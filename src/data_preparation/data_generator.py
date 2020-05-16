import parameters
from data_preparation.datasynthesis import generate_word_img
import cv2


def generate_images(text_file, save_directory, method):
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        i = 0
        generated = 0
        failed = 0
        csv_file = open(parameters.generated_csv_path, 'w', encoding='utf-8')
        for line in lines:

            # Generate image
            try:
                word = line.replace('\n', '')
                image = generate_word_img(method, word)
                image_name = str(i) + '.jpg'
                csv_file.write(image_name + ',' + word + '\n')
                cv2.imwrite(f'{save_directory}/{i}.jpg', image)
                generated += 1
                print(f'Generated {generated} images')

            except KeyError:
                failed += 1
                print(f'Failed to generate {failed} images')

            print(f'Processed word {i}')
            i += 1


if __name__ == '__main__':
    generate_images(parameters.bangla_words_list, parameters.generated_images_save_dir,
                    parameters.generation_method)





