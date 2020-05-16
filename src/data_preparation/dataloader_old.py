import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import parameters
import math
import cv2


class DataSetOCROld(Dataset):

    def __init__(self, csv_file_path, root_directory):
        self.data = pd.read_csv(csv_file_path, sep=',', header=None, encoding='utf-8')
        self.max_image_width = parameters.max_image_width
        self.max_image_height = parameters.max_image_height
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.root = root_directory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image
        word = self.data.iloc[index, 1]
        image_name = self.data.iloc[index, 0]
        image_path = self.root + image_name

        image = cv2.imread(image_path)

        image_height = image.shape[0]
        image_width = image.shape[1]

        # Calculate image height and width requirements
        if image_height > self.max_image_height:
            image = cv2.resize(image, (image_width, 56), Image.BILINEAR)
        delta_height = 0
        if image_height < self.max_image_height:
            delta_height = self.max_image_height - image_height
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
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

        return image, word



