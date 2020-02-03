import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import features
import numpy as np
import cv2
from io import BytesIO
import parameters


def load_file(path_name: str) -> list:
    with open(path_name, encoding='utf-8') as file:
        lines = file.readlines()
    return lines


def line_to_word(line: str) -> list:
    words = line.split()
    return words


def word_count(line: str) -> int:
    count = len(line_to_word(line))
    return count


def char_count(line: str) -> int:
    return len(line)


def remove_duplicate_word(word_list: list) -> list:
    words = list(set(word_list))
    return words


def show_image(image: Image):
    plt.imshow(image)
    plt.show()


def load_font(font_filename: str, font_size: int) -> object:
    file = open(parameters.font_files_root_path + font_filename, "rb")
    #file=open(font_filename,"rb")
    #bytes_font = BytesIO(file.read())
    #font = ImageFont.truetype(bytes_font, font_size)
    #font = ImageFont.truetype(font=bytes_font, size=font_size, encoding='unic',layout_engine=None)
    font = ImageFont.truetype(file, font_size, encoding="unic")
    return font


def image_size_define(font_size: int, num_char: int, num_lines: int) -> Image:
    height = font_size * (num_lines + 1)
    width = font_size * int(num_char) + 10
    image = Image.new("RGB", (width, height))

    return image


def text_to_image(text: str, font_size: int, font: object, x: int, y: int) -> Image:
    char_number = char_count(text)
    # print(char_number)
    line_number = 1
    image = image_size_define(font_size, char_number, line_number)

    canvas = ImageDraw.Draw(image)
    canvas.text((x, y), text, font=font)
    return image


def save_image(path: str, image: Image):
    image_name = path
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((20, 20), np.uint8)
    image_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # find contours
    contour, hier = cv2.findContours(image_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    sorted_contour = sorted(contour, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_contour):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI (croping x,y)
        # roi = image[y:y + h, x:x + w]
        crop_image = thresh[y:y + h, x:x + w]
        cv2.imwrite(image_name, crop_image)
    # Image.Image.save(image, image_name)


def process_image(image: Image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((20, 20), np.uint8)
    image_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # find contours
    contour, _ = cv2.findContours(image_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    sorted_contour = sorted(contour, key=lambda key: cv2.boundingRect(key)[0])
    for i, ctr in enumerate(sorted_contour):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI (croping x,y)
        # roi = image[y:y + h, x:x + w]
        crop_image = thresh[y:y + h, x:x + w]
        return crop_image


def read_font_name(file_name: str) -> list:
    with open(file_name) as file:
        lines = file.read()
    names = lines.splitlines()
    return names
