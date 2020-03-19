import torch
import torchvision.transforms as transforms
import math
from data_preparation.ImageAugmentation import ImageAugmentation
import cv2
from PIL import Image
import torchvision.transforms as transforms


def my_collate(batch):
    images, labels = zip(*batch)
    max_width = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_widths = []
    for image in images:
        image_widths.append(math.ceil(image.size[0] / 16))
        if image.size[0] > max_width:
            max_width = image.size[0]
    final_images = []
    for image in images:
        delta_width = max_width - image.size[0]
        pad = transforms.Pad((0, 0, delta_width, 0), fill=0, padding_mode='constant')
        image = pad(image)
        image = transform(image)
        final_images.append(image)

    images = torch.cat([t.unsqueeze(0) for t in final_images], 0)

    image_widths = torch.IntTensor(image_widths)

    # print('max width {}'.format(max_width))
    return images, labels, image_widths


def old_collate(batch):
    images, labels = zip(*batch)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    final_images = []
    # img_aug = ImageAugmentation(0.2, 0.2, 0.2, 0.2, list(images))
    # augmented_images = img_aug.select_image()

    augmented_images = list(images)
    for image in augmented_images:
        # print(image.shape)
        image = Image.fromarray(image).convert('RGB')
        img = transform(image)
        final_images.append(img)

    images = tuple(final_images)

    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    return images, labels


def collate_v3(batch):
    images, labels = zip(*batch)
    images = torch.cat([t.unsqueeze(0) for t in images], 0)
    return images, labels


def loadData(v, data):
    v.resize_(data.size()).copy_(data)

