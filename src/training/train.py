from data_preparation.dataloader import DataSetOCR
from utills.dataloader_services import *
from torch.utils.data import DataLoader
import parameters
from models import crnn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from utills.string_label_converter import averager, StrLabelConverter
import torch.nn.functional as F
import nltk
from models import lenet_lstm

train_dataset = DataSetOCR(
    csv_file_path= parameters.train_csv_path,
    text_file_path= parameters.text_file_path,
    root_directory= parameters.train_root)
assert train_dataset

test_dataset = DataSetOCR(
    csv_file_path= parameters.test_csv_path,
    text_file_path= parameters.text_file_path,
    root_directory= parameters.test_root)
assert test_dataset

dataloader_params = {
    'batch_size': 10,
    'shuffle': True,
    'collate_fn': my_collate,
    'drop_last': True

}

train_loader = DataLoader(train_dataset, **dataloader_params)
train_iter = iter(train_loader)

torch.backends.cudnn.benchmark = True

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(parameters.max_image_height, 3, parameters.number_of_classes, 256)
crnn.apply(weights_init)
criterion = CTCLoss(zero_infinity=True)
optimizer = optim.Adam(crnn.parameters(), lr=0.001, betas=(0.5, 0.9))
loss_avg = averager()

image = torch.FloatTensor(dataloader_params['batch_size'], 3, parameters.max_image_width, parameters.max_image_height)
text = torch.IntTensor(dataloader_params['batch_size'] * 5)
length = torch.IntTensor(dataloader_params['batch_size'])

if torch.cuda.is_available():
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(1))
    image = image.cuda()
    text = text.cuda()
    length = length.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)
string_converter = StrLabelConverter()
# string_converter.convert_integer_to_string()

def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    # Loading validation data
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=10, collate_fn=my_collate, drop_last= True)
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    edit_distance = 0
    loss_avg = averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        images, texts, preds_actual_length = data
        batch_size = images.size(0)
        loadData(image, images)
        t, l = string_converter.convert_string_to_integer(texts, [])
        loadData(text, t)
        loadData(length, l)

        preds = crnn(image)
        preds = F.log_softmax(preds, 2)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        # print('targets {}'.format(length))

        cost = criterion(preds, text, preds_actual_length, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(1)  # originial 2

        preds = preds.transpose(1, 0).contiguous().view(-1)

        sim_preds = string_converter.convert_integer_to_string(preds.data, preds_actual_length.data)

        cpu_texts = string_converter.convert_integer_to_string(text, length)

        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1
            edit_distance += nltk.edit_distance(pred, target)
            print('Prediction : {}   GT : {}'.format(pred, target))
    mean_edit_distance = edit_distance/float(max_iter * 10)
    accuracy = (n_correct / float(max_iter * 10))*100
    print('Test loss: %f, accuracy: %f , Mean Edit Distance : %f' % (loss_avg.val(), accuracy, mean_edit_distance))


def trainBatch(model, train_iter, criterion, optimizer):
    try:
        data = train_iter.next()
    except StopIteration:
        train_iter = iter(train_loader)
        data = train_iter.next()

    images, texts, preds_actual_length = data
    loadData(image, images)
    t, l = string_converter.convert_string_to_integer(texts, [])
    loadData(text, t)
    loadData(length, l)
    batch_size = dataloader_params['batch_size']
    optimizer.zero_grad()
    # print('image size {}'.format(image.size()))
    preds = model(image)
    preds = F.log_softmax(preds, 2)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_actual_length, length)/batch_size
    # print(cost)
    # print(batch_size)
    cost.backward()
    optimizer.step()
    # print(cost)
    return cost


for epoch in range(10):
    i = 0

    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, train_iter, criterion, optimizer)
        # print('from train batch {}'.format(cost))
        loss_avg.add(cost)
        i += 1

        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss: %f' % (epoch, 10, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % 50 == 0:
            val(crnn, test_dataset, criterion)

        #do checkpointing
        # if i % 50 == 0:
        #     torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format('/home/bjit/ocr/weights/', epoch, i))
    # torch.save(crnn.state_dict(), '{0}/lenet_lstm_{1}.pth'.format('/home/bjit/ocr/weights/', epoch))
