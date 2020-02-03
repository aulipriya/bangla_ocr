import pyvips
from utills import data_generation_services as util
import parameters
import random
import numpy as np
import cv2


def vips2numpy(vi):
    format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.uint8,  #np.int16
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


def word_to_image(word):
    font_names = util.read_font_name(parameters.font_names_list_path)
    font_name = random.choice(font_names)
    max_w = 2000
    image = pyvips.Image.text(word,
                              font=font_name,
                              width=max_w,
                              dpi=250)
    image = vips2numpy(image)
    kernel = np.ones((20, 20), np.uint8)
    image_dilation = cv2.dilate(image, kernel, iterations=1)
    # find contours
    contour, _ = cv2.findContours(image_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    sorted_contour = sorted(contour, key=lambda key: cv2.boundingRect(key)[0])
    crop_image = image
    for i, ctr in enumerate(sorted_contour):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI (croping x,y)
        # roi = image[y:y + h, x:x + w]
        crop_image = image[y:y + h, x:x + w]
    return crop_image


def main():
    num = 1
    with open(parameters.train_text_file_path, "r", encoding='UTF-8') as filestream:
        for text in filestream.readlines():
            image = word_to_image(text)
            cv2.imwrite(f'../printed_data/{num}.png', image)
            num += 1

