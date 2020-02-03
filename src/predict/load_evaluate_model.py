from data_preparation.dataloader import DataSetOCR
from utills.dataloader_services import *
from torch.utils.data import DataLoader
import parameters
from models import crnn
from models.mobilent_rnn import MobilenetRNN
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from utills.string_label_converter import averager, StrLabelConverter
import torch.nn.functional as F
import nltk

test_dataset = DataSetOCR(
    csv_file_path='/home/aulipriya/Desktop/fake_sythetic/fake_synthetic.csv',
    root_directory='/home/aulipriya/Desktop/fake_sythetic/top_hat_5_5/')
assert test_dataset
batch = 2
dataloader_params = {
    'batch_size': batch,
    'shuffle': True,
    'collate_fn': old_collate,
    'drop_last': True

}

test_loader = DataLoader(test_dataset, **dataloader_params)

PATH = '../weights/hand_print_mobilenet_v2_24.pth'

# crnn = crnn.CRNN(parameters.max_image_height, 3, parameters.number_of_classes, 256)
crnn = MobilenetRNN(parameters.input_channels, parameters.number_of_classes, 256)
crnn = torch.nn.DataParallel(crnn)
crnn.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
crnn.eval()

image = torch.FloatTensor(dataloader_params['batch_size'], 3, parameters.max_image_width, parameters.max_image_height)
text = torch.IntTensor(dataloader_params['batch_size'] * 5)
length = torch.IntTensor(dataloader_params['batch_size'])

with torch.no_grad():
    string_converter = StrLabelConverter()
    loss_function = CTCLoss(zero_infinity=True)
    total_cost = 0
    if torch.cuda.is_available():
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(1))
        image = image.cuda()
        text = text.cuda()
        length = length.cuda()

    counter = 0
    n_correct = 0
    for images, labels in test_loader:
        counter = counter + 1
        batch_size = images.size(0)
        loadData(image, images)
        integer_labels, label_lengths = string_converter.convert_string_to_integer(labels, [])
        loadData(text, integer_labels)
        loadData(length, label_lengths)

        output = crnn(image)
        output = F.log_softmax(output, 2)
        output_size = Variable(torch.IntTensor([output.size(0)] * batch_size))
        cost = loss_function(output, text, output_size, length) / batch_size
        total_cost = total_cost + cost.item()
        _, output = output.max(2)
        output = output.transpose(1, 0).contiguous().view(-1)
        predicted_texts = string_converter.convert_integer_to_string(output.data, output_size.data)
        ground_truth_texts = string_converter.convert_integer_to_string(text.data, length.data)
        for pred, target in zip(predicted_texts, ground_truth_texts):
            if pred == target:
                n_correct += 1
            print('Prediction : {}   GT : {}'.format(pred, target))

    average_cost = total_cost / float(counter)
    accuracy = n_correct / (batch * counter) * 100

    print('Loss {}'.format(average_cost))
    print('Accuracy {}'.format(accuracy))





