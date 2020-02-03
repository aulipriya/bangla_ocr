from utills import data_generation_services as util
import random
import parameters


FILE_PATH = '../asset/nan_test.txt'
FONT_SIZE = 50
SAVE_DIR = '../printed_data'
LABEL_FILE_PATH = '../asset/number_label.txt'
# DATA_FILE = open('/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/src/data_preparation/test_font/number_label.csv', 'w')
FONT_NAME_PATH =parameters.font_names_list_path
IMAGE_START_X, IMAGE_START_Y = 20, 20


def main():
    line = util.load_file(FILE_PATH)
    words = []
    for i in range(0, len(line)):
        line_words = util.line_to_word(line[i])
        string_word = ''.join(line_words)
        words.append(string_word)

    font_names = util.read_font_name(FONT_NAME_PATH)
    words = util.remove_duplicate_word(words)
    #label = open(LABEL_FILE_PATH, 'w')
    label = open(LABEL_FILE_PATH, "w", encoding='utf-8')
    for i in range(0, len(words)):
        font_name = random.choice(font_names)
        image_name = str(SAVE_DIR) + '/' + str(words[i]) + '_promona.jpg'
        csv_image_name = str(words[i]) + '_promona.jpg'
        font = util.load_font(font_name, FONT_SIZE)
        final_image = util.text_to_image(words[i], FONT_SIZE, font, IMAGE_START_X, IMAGE_START_Y)
        util.save_image(image_name, final_image)
        #DATA_FILE.write(str(csv_image_name) + ',' + str(i) + '\n')

        label.write(str(words[i]) + '\n')
        print('done word no : ', i)

        # if i == 3:
        #     label.close()
        #     break


def printed_seq_to_img(word):
    font_names = util.read_font_name(FONT_NAME_PATH)
    font_name = random.choice(font_names)
    font = util.load_font(font_name, FONT_SIZE)
    final_image = util.text_to_image(word, FONT_SIZE, font, IMAGE_START_X, IMAGE_START_Y)
    final_image = util.process_image(final_image)
    return final_image

