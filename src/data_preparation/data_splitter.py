import os
import numpy as np
import shutil
import ntpath
import re
import pandas as pd


def split_data_directories():
    # # Creating Train / Val / Test folders (One time use)
    root_dir = 'E:/Work/bangla_ocr_version_2/data/word_data/'

    # Creating directories
    os.makedirs(root_dir + 'train')
    os.makedirs(root_dir + 'val')
    os.makedirs(root_dir + 'test')

    # Creating label-to-line-number csv files
    train_label_file = open(root_dir + 'train/train_labels.csv', 'w')
    test_label_file = open(root_dir + 'test/test_labels.csv', 'w')
    validation_label_file = open(root_dir + 'val/val_labels.csv', 'w')

    # Creating partitions of the data after shuffling
    src = root_dir + 'images'  # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * 0.7),
                                                               int(len(allFileNames) * 0.85)])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images and creating csv label files for each directory
    for name in train_FileNames:
        shutil.copy(name, root_dir + "train")
        csv_image_name = ntpath.basename(name)
        line_number = re.findall('\d+', csv_image_name)
        train_label_file.write(str(csv_image_name) + ',' + str(line_number[0]) + '\n')

    for name in val_FileNames:
        shutil.copy(name, root_dir + "val")
        csv_image_name = ntpath.basename(name)
        line_number = re.findall('\d+', csv_image_name)
        validation_label_file.write(str(csv_image_name) + ',' + str(line_number[0]) + '\n')

    for name in test_FileNames:
        shutil.copy(name, root_dir + "test")
        csv_image_name = ntpath.basename(name)
        line_number = re.findall('\d+', csv_image_name)
        test_label_file.write(str(csv_image_name) + ',' + str(line_number[0]) + '\n')


def split_data_text_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        total_word_list = []
        for line in lines:
            line = line.replace('\n', '')
            total_word_list.append(line)
        np.random.shuffle(total_word_list)
        train_words, val_words, test_words = np.split(np.array(total_word_list),
                                                      [int(len(total_word_list) * 0.7),
                                                       int(len(total_word_list) * 0.85)])
        train_file = open('train_label.txt', 'w')
        for word in train_words:
            train_file.write(word + '\n')
        validation_file = open('val_label.txt', 'w')
        for word in val_words:
            validation_file.write(word + '\n')
        test_file = open('test_label.txt', 'w')
        for word in test_words:
            test_file.write(word + '\n')


def split_data_from_csv_file(file_path, root_directory, train_directory, val_directory, test_directory):
    data = pd.read_csv(file_path, header=None)
    word_count = 0
    word_list = []
    for i in range(len(data)):
        if word_count < 500000:
            line = data.iloc[word_count, 1]
            word_list.append(line.replace('\n', ''))
            word_count += 10
    np.random.shuffle(word_list)
    train_words, val_words, test_words = np.split(np.array(word_list),
                                                  [int(len(word_list) * 0.7),
                                                   int(len(word_list) * 0.85)])

    train_csv = open('/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/bjit/ocr_handwritten_splits/handwritten_train.csv', 'w', encoding='utf-8')
    for word in train_words:
        rows = data.loc[data[1] == word]
        for i in range(len(rows)):
            image = rows.iloc[i, 0]
            word = rows.iloc[i, 1]
            train_csv.write(image + ',' + word + '\n')
            shutil.copy(root_directory + '/' + image, train_directory)

    val_csv = open('/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/bjit/ocr_handwritten_splits/handwritten_val.csv', 'w', encoding='utf-8')
    for word in val_words:
        rows = data.loc[data[1] == word]
        for i in range(len(rows)):
            image = rows.iloc[i, 0]
            word = rows.iloc[i, 1]
            val_csv.write(image + ',' + word + '\n')
            shutil.copy(root_directory + '/' + image, val_directory)

    test_csv = open('/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/bjit/ocr_handwritten_splits/handwritten_test.csv', 'w', encoding='utf-8')
    for word in test_words:
        rows = data.loc[data[1] == word]
        for i in range(len(rows)):
            image = rows.iloc[i, 0]
            word = rows.iloc[i, 1]
            test_csv.write(image + ',' + word + '\n')
            shutil.copy(root_directory + '/' + image, test_directory)


split_data_from_csv_file('../asset/handwritten_test.csv',
                         '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/Test_Handwritten/',
                        '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/bjit/ocr_handwritten_splits/train',
                        '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/bjit/ocr_handwritten_splits/val',
                        '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/bjit/ocr_handwritten_splits/test')
# split_data_text_file('/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/word_label.txt')







