from PIL import Image
import os
import io
import cv2
from data_preparation.handwritten_text_generation import make_character_to_folder_map_dict


def image_resizer_general():
    directory = '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/Banglalekha-processed'
    for sub_dir in os.listdir(directory):
        if sub_dir in ['3', '4', '5', '6', '9', '11', '22', '23', '85', '86', '87',
                       '88', '89', '90', '91', '92', '93', '94', '95']:
            continue
        new_dir = '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/Banglalekha_resized_2/'+ sub_dir
        os.makedirs(new_dir)
        for image_name in os.listdir(directory + '/' + sub_dir):
            try:
                image = Image.open(directory+'/' + sub_dir+'/' + image_name)
                image = image.resize((28, 28), Image.ANTIALIAS)
                image.save(new_dir+'/'+image_name, 'png', quality=90)
            except:
                continue


def image_resizer_for_extra_top(character_list):
    directory_map = make_character_to_folder_map_dict('/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/BanglaLekha-Isolated/bangla_characters_list.txt')
    for character in character_list:
        image_folder = str(directory_map[character])
        new_directory = '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/Banglalekha_resized_2/' + image_folder
        os.makedirs(new_directory)
        root_path = '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/Banglalekha-processed'
        directory_path = root_path + '/' + image_folder
        for image_name in os.listdir(directory_path):
            try:
                image = Image.open(directory_path + '/' + image_name)
                image = image.resize((28, 42), Image.ANTIALIAS)
                image.save(new_directory + '/' + image_name, 'png', quality=90)
            except:
                print(image_folder)
                print(image_name)
                continue


def image_resizer_for_extra_top_width(character_list):
    directory_map = make_character_to_folder_map_dict('/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/BanglaLekha-Isolated/bangla_characters_list.txt')
    for character in character_list:
        image_folder = str(directory_map[character])
        new_directory = '/home/aulipriya/Downloads/Resized/' + image_folder
        os.makedirs(new_directory)
        root_path = '/home/aulipriya/Downloads/BanglaLekha-Isolated/Images'
        directory_path = root_path + '/' + image_folder
        for image_name in os.listdir(directory_path):
            image = Image.open(directory_path + '/' + image_name)
            image = image.resize((42, 42), Image.ANTIALIAS)
            image.save(new_directory + '/' + image_name, 'png', quality=90)


def modifier_image_resizer():
    directory = '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/ocr/ocr-project/data/modifier-processed'
    for folder in os.listdir(directory):
        images = os.listdir(os.path.join(directory, folder))

        folder_path = os.path.join(directory, folder)
        image = Image.open(os.path.join(folder_path, 'processed.png'))
        if str(folder) == '96':
            image = image.resize((14, 28), Image.ANTIALIAS)
        elif str(folder) == '97':
            image = image.resize((14, 14), Image.ANTIALIAS)
        if str(folder) == '85':
            image = image.resize((14, 28), Image.ANTIALIAS)
        elif str(folder) == '86':
            image = image.resize((42, 42), Image.ANTIALIAS)
        elif str(folder) == '87':
            image = image.resize((42, 42), Image.ANTIALIAS)
        elif str(folder) == '88':
            image = image.resize((14, 14), Image.ANTIALIAS)
        elif str(folder) == '89':
            image = image.resize((14, 14), Image.ANTIALIAS)
        elif str(folder) == '90':
            image = image.resize((14, 14), Image.ANTIALIAS)
        elif str(folder) == '91':
            image = image.resize((14, 28), Image.ANTIALIAS)
        elif str(folder) == '92':
            image = image.resize((14, 28), Image.ANTIALIAS)
        elif str(folder) == '93':
            image = image.resize((56, 28), Image.ANTIALIAS)
        elif str(folder) == '94':
            image = image.resize((56, 42), Image.ANTIALIAS)
        elif str(folder) == '95':
            image = image.resize((28, 28), Image.ANTIALIAS)

        image.save(os.path.join(folder_path, 'modified.png'), 'png', quality=90)


# image_resizer_general()


# image_resizer_for_extra_top(['ই', 'ঈ', 'উ', 'ঊ', 'ঐ', 'ঔ', 'ট', 'ঠ'])
# image_resizer_for_extra_top(['ন্ঠ'])

# image_resizer_for_extra_top_width(['ঊ'])

modifier_image_resizer()

