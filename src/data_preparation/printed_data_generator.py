from utills import data_generation_services as util
import random
import parameters


FONT_SIZE = 50
FONT_NAME_PATH = parameters.font_names_list_path
IMAGE_START_X, IMAGE_START_Y = 20, 20


def printed_seq_to_img(word):
    font_names = util.read_font_name(FONT_NAME_PATH)
    font_name = random.choice(font_names)
    font = util.load_font(font_name, FONT_SIZE)
    final_image = util.text_to_image(word, FONT_SIZE, font, IMAGE_START_X, IMAGE_START_Y)
    final_image = util.process_image(final_image)
    return final_image
