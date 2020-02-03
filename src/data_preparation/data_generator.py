from utills import data_generation_services as util
import random
import parameters
from data_preparation.datasynthesis import generate_word_img
import cv2


def generate_images(text_file, save_directory, method):
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        i = 0
        # generation_method = ['handwritten', 'printed']
        generated = 0
        failed = 0
        csv_file = open('../asset/handwritten_labels.csv', 'w')
        for line in lines:
            # method = random.choices(population=generation_method, weights=[0.75, 0.25], k=1)[0]

            # Generate image
            try:
                image = generate_word_img(method, line.replace('\n', ''))
                cv2.imwrite(f'{save_directory}/{i}.jpg', image)
                image_name = str(i) + '.jpg'
                csv_file.write(image_name + ',' + str(i) + '\n')
                generated += 1
                print(f'Generated {generated} images')

            except KeyError:
                # image = generate_word_img('printed', line.replace('\n', ''))
                failed += 1
                print(f'Failed to generate {failed} images')

            print(f'Processed word {i}')
            i += 1


generate_images('../asset/all_words.txt', '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/OCR_Handwritten',
                'handwritten')