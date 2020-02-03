import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import parameters
import math


class DataSetOCROld(Dataset):

    def __init__(self, csv_file_path, text_file_path, root_directory):
        self.data = pd.read_csv(csv_file_path, header=None)
        self.text_file_path = text_file_path
        self.max_image_width = parameters.max_image_width
        self.max_image_height = parameters.max_image_height
        self.transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.root_directory = root_directory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.root_directory + self.data.iloc[index, 0]
        image = Image.open(path).convert('RGB')
        image_height = image.size[1]
        image_width = image.size[0]
        height_resize_flag = False
        width_resize_flag = False

        # Calculate height and width that must be padded to match max size
        if self.max_image_height > image_height:
            delta_height = self.max_image_height - image_height
        else:
            delta_height = 0
            height_resize_flag = True
        if self.max_image_width > image_width:
            delta_width = self.max_image_width - image_width
        else:
            delta_width = 0
            width_resize_flag = True
        # Calculate top and bottom padding
        if delta_height % 2 == 0:
            top = bottom = int(delta_height / 2)
        else:
            top = math.floor(delta_height / 2) + 1
            bottom = math.floor(delta_height / 2)

        # Calculate left and right padding
        if delta_width % 2 == 0:
            left = right = int(delta_width / 2)
        else:
            left = math.floor(delta_width / 2) + 1
            right = math.floor(delta_width / 2)

        # Pad image
        pad = transforms.Pad((left, top, right, bottom), fill=0, padding_mode='constant')
        padded_image = pad(image)

        # Set default resize size
        resize_width = padded_image.size[0]
        resize_height = padded_image.size[1]
        # Set resize size if needed
        if height_resize_flag:
            resize_height = self.max_image_height
        if width_resize_flag:
            resize_width = self.max_image_width

        # Resize image
        resized_image = padded_image.resize((resize_width, resize_height), Image.BILINEAR)
        resized_image = self.transform(resized_image)

        # Retrieve integer label of the image from csv file
        integer_label = self.data.iloc[index, 1:]

        # Retrieve text label of the image from the text file
        fp = open(self.text_file_path, encoding='utf8')
        lines = fp.readlines()
        text_label = lines[int(integer_label)]
        fp.close()
        return resized_image, text_label



