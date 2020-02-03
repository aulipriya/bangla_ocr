from data_preparation.handwritten_text_generation import seq_to_img
from data_preparation.printed_text_generator_2 import word_to_image


def generate_word_img(method, word):
    if method == 'handwritten':
        return seq_to_img(word)
    else:
        return word_to_image(word)

