from data_preparation.dataloader_old import DataSetOCROld
from utills.dataloader_services import *
from torch.utils.data import DataLoader
import parameters
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import CTCLoss
from utills.string_label_converter import averager, StrLabelConverter
import torch.nn.functional as F
from utills.thread_with_return_value import ThreadWithReturnValue
from models.mobilent_rnn import MobilenetRNN
import numpy as np
import matplotlib.pyplot as plt


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        #m.weight.data.normal_(0.0, 0.02)
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
    image = torch.FloatTensor(parameters.batch_size, 3, parameters.max_image_width, parameters.max_image_height)
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
        cost_function = cost_function.cuda()

    batch_size = parameters.batch_size
    optimizer_function.zero_grad()
    preds = model(image)
    preds = F.log_softmax(preds, 2)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    if torch.cuda.is_available():
        preds_size.cuda()
    iteration_cost = cost_function(preds, text, preds_size, length) / batch_size
    iteration_cost.backward()
    optimizer_function.step()
    print('Model training ended for iteration', str(i))

    return iteration_cost


def feed_data_into_model_for_validation(model, loaded_data, cost_function, optimizer_function, i):
    print('Model validation started for iteration', str(i))

    image = torch.FloatTensor(parameters.batch_size, 3, parameters.max_image_width, parameters.max_image_height)
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
    return return_results


def val(net, dataset, criterion, optimizer, max_iter=2):
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
    loss_avg = averager()
    batch_size = parameters.batch_size

    # Set number of iterations the validation function will work
    max_iter = min(max_iter, len(val_data_loader))

    # Load data for the first iteration
    data = load_data_for_each_iteration(val_iter, i, val_data_loader)
    for i in range(max_iter):

        # Define validation and data loading thread
        validation_thread = ThreadWithReturnValue(target=feed_data_into_model_for_validation, args=(net, data, criterion,
                                                                                     optimizer, i,))

        loading_thread = ThreadWithReturnValue(target=load_data_for_each_iteration, args=(val_iter,
                                                                                          i + 1, val_data_loader))
        # Start validation then loading thread for next iteration
        validation_thread.start()
        loading_thread.start()
        # Wait for validation thread to finish calculation
        results = validation_thread.join()
        cost = results['cost']
        preds = results['predictions']


        # Add validation cost
        loss_avg.add(cost)

        # Initialize converter
        string_converter = StrLabelConverter()

        # Convert predictions to texts
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        _, preds = preds.max(2)
        preds = preds.squeeze(1)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = string_converter.convert_integer_to_string(preds.data, preds_size.data)

        # Convert true labels to texts
        _, texts = data
        text = torch.IntTensor(parameters.batch_size * 5)
        length = torch.IntTensor(parameters.batch_size)
        t, l = string_converter.convert_string_to_integer(texts, [])
        loadData(text, t)
        loadData(length, l)
        if torch.cuda.is_available():
            text = text.cuda()
            length = length.cuda()

        cpu_texts = string_converter.convert_integer_to_string(text, length)

        # Wait for loading thread to load data for next iteration

        data = loading_thread.join()

        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1
            print('Prediction : {}   GT : {}'.format(pred, target))
        i += 1

    accuracy = (n_correct / float(max_iter * batch_size))*100
    return loss_avg.val(), accuracy


def main():
    # Define datasets
    train_dataset = DataSetOCROld(csv_file_path=parameters.train_text_file_path,
                                  root_directory=parameters.train_root_directory)

    print("train_dataset_loading complete")
    assert train_dataset

    validation_dataset = DataSetOCROld(csv_file_path=parameters.validation_text_file_path,
                                       root_directory=parameters.val_root_directory)
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
    model = MobilenetRNN(parameters.input_channels, parameters.number_of_classes, 256)
    criterion = CTCLoss(zero_infinity=True)

    optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate, betas=(0.5, 0.9))
    loss_avg = averager()

    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(1))
    training_details = open(parameters.training_details_file, 'w')
    validation_details = open(parameters.validation_details_file, 'w')
    epoch_cost = 0
    training_cost = dict()
    validation_cost = dict()
    validation_accuracy = dict()
    for epoch in range(1, parameters.epochs):
        print(f'Starting for epoch {epoch}')
        iteration = 1
        train_iter = iter(train_loader)
        data = load_data_for_each_iteration(train_iter, iteration, train_loader)

        while iteration <= len(train_loader):
            for p in model.parameters():
                p.requires_grad = True
            model.train()
            training_thread = ThreadWithReturnValue(target=feed_data_into_model, args=(model, data, criterion,
                                                                                       optimizer, iteration,))

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
            epoch_cost += cost
            iteration += 1
            if iteration % parameters.display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' % (epoch, parameters.epochs, iteration, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if iteration % parameters.validation_interval == 0:
                val_cost, accuracy = val(model, validation_dataset, criterion, optimizer)
                validation_details.write(f'Epoch ---- {epoch} iteration --- {iteration} cost --- {val_cost} accuracy --- {accuracy} \n')
                validation_cost[epoch] = val_cost
                validation_accuracy[epoch] = validation_accuracy

        print(f'Ending for epoch {epoch}')
        training_details.write(f'Epoch {epoch}     average_cost : {epoch_cost} \n')
        training_cost[epoch] = epoch_cost / len(train_loader)
        print(epoch_cost)

        torch.save(model.state_dict(), '{0}/hand_print_mobilenet_1{1}.pth'.format(parameters.weights_folder, epoch))
    return training_cost, validation_cost, validation_accuracy


if __name__ == '__main__':
    training_loss, validation_loss, validation_accuracy = main()
    training_epochs = np.array(list(training_loss.keys()))
    training_losses = np.array(list(training_loss.values()))
    plt.plot(training_epochs, training_losses, 'b')
    validation_loss_epochs = np.array(list(validation_loss.keys()))
    validation_losses = np.array(list(validation_loss.values()))
    plt.plot(validation_loss_epochs, validation_losses, 'r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses ')
    # show a legend on the plot
    plt.legend()
    plt.savefig('loss')

