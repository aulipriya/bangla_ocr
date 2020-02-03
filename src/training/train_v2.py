#from .data_preparation.dataloader_v2 import DataSetV2
from data_preparation.dataloader_v2 import DataSetV2
from utills.dataloader_services import *
from torch.utils.data import DataLoader
import parameters
from models.crnn import CRNN
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from utills.string_label_converter import averager, StrLabelConverter
import torch.nn.functional as F
from utills.thread_with_return_value import ThreadWithReturnValue
from models.mobilent_rnn import MobilenetRNN
import nltk
import numpy as np


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Function for loading data
def load_data_for_each_iteration(train_iterable, i, train_loader):
    print('Data Loading started for iteration', str(i))
    try:
        loaded_data = train_iterable.next()
    except StopIteration:
        train_iterable = iter(train_loader)
        loaded_data = train_iterable.next()
    print('Data Loading ended for iteration', str(i))
    return loaded_data


# Function to feed data to model and calculate cost
def feed_data_into_model(model, loaded_data, cost_function, optimizer_function, i):
    print('Model training started for iteration', str(i))
    # Initializing image, text and length variables
    image = torch.FloatTensor(parameters.batch_size, 3, parameters.max_image_width, parameters.desired_height)
    text = torch.IntTensor(parameters.batch_size * 5)
    length = torch.IntTensor(parameters.batch_size)

    images, texts = loaded_data
    # print(f'grounf truths -- {texts}')
    loadData(image, images)
    string_converter = StrLabelConverter()
    t, l = string_converter.convert_string_to_integer(texts, [])
    # print(f'ground truth integer labels -- {t}')
    loadData(text, t)
    loadData(length, l)
    if torch.cuda.is_available():
        image = image.cuda()
        # text = text.cuda()
        length = length.cuda()
        cost_function = cost_function.cuda()
    numpy_image = image.numpy()[0]
    # print(f'image -- {numpy_image.shape}')
    # non_minus = list()
    # non_blank_file = open('../asset/non_blank.txt', 'w')
    # for i in range(numpy_image.shape[0]):
    #     for j in range(numpy_image.shape[1]):
    #         for k in range(numpy_image.shape[2]):
    #             if numpy_image[i][j][k] != -1:
    #                 non_minus.append(numpy_image[i][j][k])
    # print(f'non-minus {non_minus}')
    # non_blank_file.write(f'iteration {i}   average_cost : {non_minus} \n')
    batch_size = parameters.batch_size
    optimizer_function.zero_grad()
    preds = model(image)
    preds = F.log_softmax(preds, 2)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    iteration_cost = cost_function(preds, text, preds_size, length) / batch_size
    print(f'cost ---{iteration_cost}')
    preds = None
    images = None
    iteration_cost.backward()
    optimizer_function.step()
    print('Model training ended for iteration', str(i))

    return iteration_cost


def feed_data_into_model_for_validation(model, loaded_data, cost_function, optimizer_function, i):
    print('Model validation started for iteration', str(i))

    image = torch.FloatTensor(parameters.batch_size, 3, parameters.max_image_width, parameters.desired_height)
    text = torch.IntTensor(parameters.batch_size * 5)
    length = torch.IntTensor(parameters.batch_size)

    images, texts = loaded_data
    loadData(image, images)
    string_converter = StrLabelConverter()
    t, l = string_converter.convert_string_to_integer(texts, [])
    loadData(text, t)
    loadData(length, l)
    if torch.cuda.is_available():
        image = image.cuda()
        # text = text.cuda()
        length = length.cuda()
        cost_function = cost_function.cuda()
    batch_size = parameters.batch_size
    optimizer_function.zero_grad()
    preds = model(image)
    preds = F.log_softmax(preds, 2)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    iteration_cost = cost_function(preds, text, preds_size, length) / batch_size
    print('Model validation ended for iteration', str(i))
    return_results = {
        'cost': iteration_cost,
        'predictions': preds
    }
    preds = None
    return return_results


def val(net, dataset, criterion, optimizer, max_iter=100):
    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    # Define dataloader for validation
    val_data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=parameters.batch_size, collate_fn=old_collate, drop_last=True)
    val_iter = iter(val_data_loader)

    # Initialize counters
    i = 0
    n_correct = 0
    edit_distance = 0
    loss_avg = averager()
    batch_size = parameters.batch_size

    # Set number of iterations the validation function will work
    max_iter = min(max_iter, len(val_data_loader))

    # Load data for the first iteration
    data = load_data_for_each_iteration(val_iter, i, val_data_loader)
    for i in range(max_iter):
        # Define validation and data loading thread
        validation_thread = ThreadWithReturnValue(target=feed_data_into_model_for_validation, args=(net, data,
                                                                                                    criterion,
                                                                                                    optimizer,
                                                                                                    i,))

        loading_thread = ThreadWithReturnValue(target=load_data_for_each_iteration, args=(val_iter,
                                                                                          i + 1,
                                                                                          val_data_loader,))
        # Start validating data
        validation_thread.start()
        loading_thread.start()
        # Wait for validation thread to finish calculation
        results = validation_thread.join()
        # results = feed_data_into_model_for_validation(net, data, criterion, optimizer, i)
        cost = results['cost']
        preds = results['predictions']
        results = None

        # Add validation cost
        loss_avg.add(cost)

        # Initialize converter
        string_converter = StrLabelConverter()

        # Convert predictions to texts
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        _, preds = preds.max(2)
        preds = preds.squeeze(1)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        # preds = preds.contiguous().view(-1)
        sim_preds = string_converter.convert_integer_to_string(preds.data, preds_size.data)
        preds = None

        # Convert true labels to texts
        _, texts = data
        data = None
        text = torch.IntTensor(parameters.batch_size * 5)
        length = torch.IntTensor(parameters.batch_size)
        t, l = string_converter.convert_string_to_integer(texts, [])
        loadData(text, t)
        loadData(length, l)
        if torch.cuda.is_available():
            text = text.cuda()
            length = length.cuda()

        cpu_texts = string_converter.convert_integer_to_string(text, length)

        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1
            edit_distance += nltk.edit_distance(pred, target)
            print('Prediction : {}   GT : {}'.format(pred, target))
        # Wait for loading thread to load data for next iteration
        data = loading_thread.join()
        i += 1
    mean_edit_distance = edit_distance/float(max_iter * batch_size)
    accuracy = (n_correct / float(max_iter * batch_size))*100
    print('Test loss: %f, accuracy: %f , Mean Edit Distance : %f' % (loss_avg.val(), accuracy, mean_edit_distance))
    return accuracy


def main():
    # Define datasets
    train_dataset = DataSetV2(text_file_path=parameters.train_text_file_path)
    assert train_dataset

    validation_dataset = DataSetV2(text_file_path=parameters.validation_text_file_path)
    assert validation_dataset

    # Define parameter for dataloader
    dataloader_params = {
        'batch_size': parameters.batch_size,
        'shuffle': True,
        'collate_fn': old_collate,
        'drop_last': True

    }

    # Define dataloader
    train_loader = DataLoader(train_dataset, **dataloader_params)

    torch.backends.cudnn.benchmark = True

    # Initialize model, optimizer, cost function and average counter
    # crnn = CRNN(parameters.max_image_height, 1, parameters.number_of_classes, 256)
    crnn = MobilenetRNN(parameters.input_channels, parameters.number_of_classes, 256)
    # crnn.apply(weights_init)
    criterion = CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(crnn.parameters(), lr=parameters.learning_rate, betas=(0.5, 0.9))
    loss_avg = averager()

    if torch.cuda.is_available():
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(1))
    training_details = open(parameters.training_details_file, 'w')
    best_val_accuracy = 0
    for epoch in range(1, 11):
        print(f'Starting for epoch {epoch}')
        iteration = 1
        train_iter = iter(train_loader)
        data = load_data_for_each_iteration(train_iter, iteration, train_loader)
        total_iterations = len(train_loader)
        while iteration <= total_iterations:
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()
            training_thread = ThreadWithReturnValue(target=feed_data_into_model, args=(crnn, data, criterion,
                                                                                       optimizer, iteration,))
            data = None

            if iteration != len(train_loader):
                loading_thread = ThreadWithReturnValue(target=load_data_for_each_iteration, args=(train_iter,
                                                                                                  iteration + 1,
                                                                                                  train_loader))
            training_thread.start()
            if iteration != len(train_loader):
                loading_thread.start()
            cost = training_thread.join()
            if iteration != len(train_loader):
                data = loading_thread.join()
            loss_avg.add(cost)
            iteration += 1
            training_details.write(f'Epoch {epoch}   iteration {iteration}   average_cost : {loss_avg.val()} \n')
            if iteration % parameters.display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' % (epoch, 10, iteration, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if iteration % parameters.validation_interval == 0:
                accuracy = val(crnn, validation_dataset, criterion, optimizer)
                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    torch.save(crnn.state_dict(), '{0}/hand_print_v2{1}_{2}_{3}.pth'.format(parameters.weights_folder,
                                                                                            epoch,
                                                                                            iteration,
                                                                                            best_val_accuracy))

        print(f'Ending for epoch {epoch}')


if __name__ == '__main__':
    main()

