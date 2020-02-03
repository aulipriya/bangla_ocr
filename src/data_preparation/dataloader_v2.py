import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from data_preparation.datasynthesis import generate_word_img
from PIL import Image
import torchvision.transforms as transforms
import parameters
import math
import random
import cv2
import numpy
from utills.dataloader_services import old_collate


class DataSetV2(Dataset):

    def __init__(self, text_file_path):
        self.data = pd.read_csv(text_file_path, sep='\n', header=None, encoding='utf-8')
        self.text_file_path = text_file_path
        self.max_image_width = parameters.max_image_width
        self.max_image_height = parameters.max_image_height
        self.transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve the word to generate
        word = self.data.iloc[index, 0]
        generation_method = ['handwritten', 'printed']
        try:
            # Select a method for image generation randomly
            method = random.choices(population=generation_method, weights=[0.75, 0.25], k=1)[0]

            # Generate image
            image = generate_word_img(method, word)

        except KeyError:
            # print(f'Cannot generate for {word}')
            image = generate_word_img('printed', word)

        image_height = image.shape[0]
        image_width = image.shape[1]

        # Calculate image height and width requirements
        if image_height > 56:
            image = cv2.resize(image, (image_width, 56), Image.BILINEAR)
        delta_height = 0
        if image_height < 56:
            delta_height = 56 - image_height
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
        # Convert image array to PIL image and apply normalization and tensor transformation

        image = Image.fromarray(image).convert('RGB')
        image = self.transform(image)
        # cv2.imshow('s', image)
        # cv2.waitKey(0)
        return image, word

