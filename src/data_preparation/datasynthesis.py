from data_preparation.handwritten_text_generation import seq_to_img
from data_preparation.printed_data_generator import printed_seq_to_img


def generate_word_img(method, word):
    if method == 'handwritten':
        return seq_to_img(word)
    else:
        return printed_seq_to_img(word)

