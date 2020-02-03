import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import parameters
import math


class DataSetOCR(Dataset):

    def __init__(self, csv_file_path, root_directory):
        self.data = pd.read_csv(csv_file_path, header=None, encoding='utf-8')
        # self.text_file_path = text_file_path
        self.max_image_width = parameters.max_image_width
        # self.max_image_height = 0
        self.transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.root_directory = root_directory
        # self.image_dict = self.proces_image()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.root_directory + self.data.iloc[index, 0]
        # image = self.image_dict[path]
        image = Image.open(path).convert('RGB')

        # Get image height and width
        image_height = image.size[1]
        image_width = image.size[0]

        # Calculate delta height and resize accordingly
        if image_height > parameters.desired_height:
            image = image.resize((image.size[0], parameters.desired_height), Image.BILINEAR)
        delta_height = 0
        if image_height < parameters.desired_height:
            delta_height = parameters.desired_height - image_height

        # Calculate delta width
        delta_width = self.max_image_width - image_width

        # Calculate left and right padding
        if delta_width % 2 == 0:
            left = right = int(delta_width / 2)
        else:
            left = math.floor(delta_width / 2) + 1
            right = math.floor(delta_width / 2)

        # Calculate top and bottom padding
        if delta_height % 2 == 0:
            top = bottom = int(delta_height / 2)
        else:
            top = math.floor(delta_height / 2) + 1
            bottom = math.floor(delta_height / 2)

        # Add padding to image
        pad = transforms.Pad((left, top, right, bottom), fill=0, padding_mode='constant')
        image = pad(image)

        # Transform image to tensor and normalize
        image = self.transform(image)
        # Retrieve integer label of the image from csv file
        text_label = self.data.iloc[index, 1]
        # print(image.shape)

        # Retrieve text label of the image from the text file
        # fp = open(self.text_file_path, encoding='utf8')
        # lines = fp.readlines()
        # text_label = lines[int(integer_label)]
        # fp.close()
        return image, text_label

    def proces_image(self):
        image_dict = {}
        for i in range(self.__len__()):
            path = self.root_directory + self.data.iloc[i, 0]

            image = Image.open(path).convert('RGB')

            # Get image height and width
            image_height = image.size[1]
            image_width = image.size[0]

            # Calculate delta height and resize accordingly
            if image_height > parameters.desired_height:
                image = image.resize((image.size[0], parameters.desired_height), Image.BILINEAR)
            delta_height = 0
            if image_height < parameters.desired_height:
                delta_height = parameters.desired_height - image_height

            # Calculate delta width
            delta_width = self.max_image_width - image_width

            # Calculate left and right padding
            if delta_width % 2 == 0:
                left = right = int(delta_width / 2)
            else:
                left = math.floor(delta_width / 2) + 1
                right = math.floor(delta_width / 2)

            # Calculate top and bottom padding
            if delta_height % 2 == 0:
                top = bottom = int(delta_height / 2)
            else:
                top = math.floor(delta_height / 2) + 1
                bottom = math.floor(delta_height / 2)

            # Add padding to image
            pad = transforms.Pad((left, top, right, bottom), fill=0, padding_mode='constant')
            image = pad(image)

            # Transform image to tensor and normalize
            image = self.transform(image)
            image_dict[path] = image
        return image_dict


# train_dataset = DataSetOCR(
#     csv_file_path= parameters.train_csv_path,
#     text_file_path= parameters.text_file_path,
#     root_directory= parameters.train_root)
#
# from utills.dataloader_services import my_collate
# dataloader_params = {
#     'batch_size': 10,
#     'shuffle': True,
#     'collate_fn': my_collate,
#     'drop_last': True
#
# }
#
# train_loader = DataLoader(train_dataset, **dataloader_params)
# train_iter = iter(train_loader)
# train_iter.next()