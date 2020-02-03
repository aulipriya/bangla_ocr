import cv2
import os
import numpy as np


def distort_opening(kernel, input_directory, save_directory):
    images = os.listdir(input_directory)
    os.makedirs(save_directory)
    for image in images:
        img = cv2.imread(os.path.join(input_directory, image))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(save_directory, image), opening)


def distort_closing(kernel, input_directory, save_directory):
    images = os.listdir(input_directory)
    os.makedirs(save_directory)
    for image in images:
        img = cv2.imread(os.path.join(input_directory, image))
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(save_directory, image), closing)


input_kernel = np.ones((7, 7), np.uint8)
distort_closing(input_kernel, '/home/aulipriya/Desktop/fake_sythetic/input', '/home/aulipriya/Desktop/fake_sythetic/closing_7_7')

