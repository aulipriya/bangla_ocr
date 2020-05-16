import os
import cv2
import random
import numpy as np
import math
import parameters


def make_character_to_folder_map_dict(text_file_path):
    bangla_character_list = []
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        for letter in lines:
            bangla_character_list.append(letter)

    bangla_character_list = bangla_character_list[1:]
    bangla_character_dict = {}
    for i, letter in enumerate(bangla_character_list):
        bangla_character_dict.update({letter: (i + 1)})
    return bangla_character_dict


def word_to_symbols(word):
    symbols = []
    i = 0
    word_len = len(word)
    is_immediate = False
    while i < word_len:
        # For compound letter i.e letters that have  '্' after them
        if i < word_len - 2 and word[i + 1] == '্':

            # For cases where there are two compound letters in a row
            if is_immediate:
                letters = symbols[len(symbols) - 1]

                # Cases where previous compound letter was a ref or jofola
                if letters in ['REF', 'JOFOLA']:
                    if word[i + 2] == 'য':
                        symbols.append('JOFOLA')
                    else:
                        symbols[len(symbols)-2] = word[i] + word[i+1] + word[i+2]

                # Cases where previous compound letter was anything other than ref or jofola
                else:
                    if word[i + 2] == 'য':
                        symbols.append('JOFOLA')
                    else:
                        symbols[len(symbols) - 1] = letters + word[i + 1] + word[i + 2]

            # For cases where there is only one compound letter
            else:
                # Append ref like a modifier after the letter
                if word[i] == 'র':
                    symbols.append(word[i+2])
                    symbols.append('REF')
                # Append jofola like a modifier after the letter
                elif word[i+2] == 'য':
                    symbols.append(word[i])
                    symbols.append('JOFOLA')
                # Other cases of compound letters ( no ref or jofola ) where the two letters must be concatenated
                else:
                    symbols.append(word[i] + word[i + 1] + word[i + 2])
            i += 2
            is_immediate = True

        # For basic letters
        else:
            if not is_immediate:
                symbols.append(word[i])
            i += 1
            is_immediate = False
    # print("Symbols_list",symbols)
    return symbols


def blend_modifier_after(letter, modifier):
    """
    blender for 'া' , 'JOFOLA'
    """
    # Taking one channel from input images

    alpha = modifier[:, :, 2]  # shape: 14, 28
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]

    # Calculating required height for images
    alpha_height_required = 56 - alpha.shape[0]
    letter_height_required = 56 - letter.shape[0]
    if alpha_height_required % 2 == 0:
        alpha_top = alpha_bottom = int(alpha_height_required / 2)
    else:
        alpha_top = math.floor(alpha_height_required / 2) + 1
        alpha_bottom = math.floor(alpha_height_required / 2)
    if letter_height_required % 2 == 0:
        letter_top = letter_bottom = int(letter_height_required / 2)
    else:
        letter_top = math.floor(letter_height_required / 2) + 1
        letter_bottom = math.floor(letter_height_required / 2)

    # Reshaping modifier to match required height
    alpha_reshaped = np.vstack((np.zeros((alpha_top, 14), np.int8),
                                alpha, np.zeros((alpha_bottom, 14), np.int8)))



    # Reshaping letter to match required height
    if letter.shape[0] > 28:
        letter_reshaped = np.vstack((letter,
                                     np.zeros((letter_height_required, 28), np.int8)))
    else:
        letter_reshaped = np.vstack((np.zeros((letter_top, 28), np.int8),
                                     letter, np.zeros((letter_bottom, 28), np.int8)))

    result = np.hstack((letter_reshaped, alpha_reshaped))
    return result


def blend_modifier_before_top(letter, modifier):
    """
    blender for 'ি', 'ৈ'
    """
    # Taking single channel from input images
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]
    alpha = modifier[:, :, 2]  # shape 42, 42 / 42, 14

    # Calculating required height for letter and modifier
    alpha_height_required = 56 - alpha.shape[0]
    letter_height_required = 56 - letter.shape[0]

    # print(alpha_height_required,letter_height_required)
    if letter_height_required % 2 == 0:
        letter_top = letter_bottom = int(letter_height_required / 2)
    else:
        letter_top = math.floor(letter_height_required / 2) + 1
        letter_bottom = math.floor(letter_height_required / 2)
    if alpha.shape[1] == 14:
        # print("alpha shape",alpha.shape)
        alpha = np.hstack((alpha, np.zeros((42, 28), np.int8)))

    # Reshaping letter and modifier
    alpha_reshaped = np.vstack((alpha, np.zeros((alpha_height_required, alpha.shape[1]), np.int8)))  # shape: 56x42
    # print("alpha_res",alpha_reshaped.shape)
    letter_width_required = 42 - letter.shape[1]
    letter_reshaped = np.hstack((np.zeros((letter.shape[0], letter_width_required), np.int8), letter))  # shape: 28x42
    # print("letter_res", letter_reshaped.shape)
    if letter_reshaped.shape[0] > 28:
        letter_reshaped = np.vstack((letter_reshaped,
                                     np.zeros((letter_height_required, letter_reshaped.shape[1]),
                                              np.int8)))  # shape: 42x42
    else:
        letter_reshaped = np.vstack((np.zeros((letter_top, letter_reshaped.shape[1]), np.int8),
                                     letter_reshaped,
                                     np.zeros((letter_top, letter_reshaped.shape[1]), np.int8)))  # shape: 42x42
    result = letter_reshaped + alpha_reshaped
    return result


def blend_modifier_before(letter, modifier):
    """
    blender for 'ে'
    """
    # Taking single channel from input images
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]
    alpha = modifier[:, :, 2]  # shape: 28x14

    # Calculating required height for modifier and symbol
    alpha_height_required = 56 - alpha.shape[0]
    letter_height_required = 56 - letter.shape[0]
    if alpha_height_required % 2 == 0:
        alpha_top = alpha_bottom = int(alpha_height_required / 2)
    else:
        alpha_top = math.floor(alpha_height_required / 2) + 1
        alpha_bottom = math.floor(alpha_height_required / 2)
    if letter_height_required % 2 == 0:
        letter_top = letter_bottom = int(letter_height_required / 2)
    else:
        letter_top = math.floor(letter_height_required / 2) + 1
        letter_bottom = math.floor(letter_height_required / 2)

    # Reshaping symbol and modifier image
    alpha_reshaped = np.vstack((np.zeros((alpha_top, alpha.shape[1]), np.int8),
                                alpha, np.zeros((alpha_bottom, alpha.shape[1]), np.int8)))  # shape: 56x14
    alpha_reshaped = np.hstack((alpha_reshaped, np.zeros((alpha_reshaped.shape[0], 28), np.int8)))

    if letter.shape[0] > 28:
        letter_reshaped = np.vstack(
            (letter, np.zeros((letter_height_required, letter.shape[1]), np.int8)))  # shape: 56x28
    else:
        letter_reshaped = np.vstack((np.zeros((letter_top, letter.shape[1]), np.int8),
                                     letter, np.zeros((letter_bottom, letter.shape[1]), np.int8)))  # shape: 56x28
    letter_reshaped = np.hstack((np.zeros((letter_reshaped.shape[0], 14), np.int8),
                                 letter_reshaped))

    # result = np.hstack((alpha_reshaped, letter_reshaped))
    # print(alpha_reshaped.shape)
    # print(letter_reshaped.shape)

    result = alpha_reshaped + letter_reshaped
    return result


def blend_modifier_around(letter, modifier_before, modifier_after):
    """
    blender for 'ো'
    """
    # Taking single channel from input images
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]
    alpha1 = modifier_before[:, :, 2]
    alpha2 = modifier_after[:, :, 2]
    # Calculating height requirements
    letter_height_required = 56 - letter.shape[0]
    if letter_height_required % 2 == 0:
        letter_top = letter_bottom = int(letter_height_required / 2)
    else:
        letter_top = math.floor(letter_height_required / 2) + 1
        letter_bottom = math.floor(letter_height_required / 2)

    # Reshaping modifier and letter
    alpha1_reshaped = np.vstack((np.zeros((14, 14), np.int8), alpha1, np.zeros((14, 14), np.int8)))  # shape: 56x14
    if letter.shape[0] > 28:
        letter_reshaped = np.vstack((letter, np.zeros((letter_height_required, letter.shape[1]), np.int8)))
    else:
        letter_reshaped = np.vstack((np.zeros((letter_top, letter.shape[1]), np.int8),
                                     letter, np.zeros((letter_bottom, letter.shape[1]), np.int8)))
    alpha2_reshaped = np.vstack((np.zeros((14, 14), np.int8), alpha2, np.zeros((14, 14), np.int8)))
    result = np.hstack((alpha1_reshaped, letter_reshaped, alpha2_reshaped))
    return result


def blend_modifier_around_top(letter, modifier):
    """
    blender for 'ৌ'
    arg letter: an image of letter read using cv2.imread unchanged, shape of 28x28
    arg modifier: a tranparant image of modifier read using cv2.imread unchanged, shape of 42x56x4
    """
    # Taking single channel from input images
    alpha = modifier[:, :, 2]  # shape 56, 42
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]

    # Calculating required height for symbol and modifier
    alpha_height_required = 56 - alpha.shape[0]
    letter_height_required = 56 - letter.shape[0]
    if letter_height_required % 2 == 0:
        letter_top = letter_bottom = int(letter_height_required / 2)
    else:
        letter_top = math.floor(letter_height_required / 2) + 1
        letter_bottom = math.floor(letter_height_required / 2)

    # Reshaping symbol and modifier
    alpha_reshaped = np.vstack((alpha, np.zeros((alpha_height_required, alpha.shape[1]), np.int8)))  # shape: 56x56

    letter_reshaped = np.hstack((np.zeros((letter.shape[0], 14), np.int8),
                                 letter, np.zeros((letter.shape[0], 14), np.int8)))  # shape: 28x56
    if letter.shape[0] > 28:
        letter_reshaped = np.vstack((letter_reshaped,
                                     np.zeros((letter_height_required, letter_reshaped.shape[1]), np.int8)))
    else:
        letter_reshaped = np.vstack((np.zeros((letter_top, letter_reshaped.shape[1]), np.int8),
                                     letter_reshaped, np.zeros((letter_bottom, letter_reshaped.shape[1]), np.int8)))
    result = letter_reshaped + alpha_reshaped
    return result


def blend_modifier_bottom(letter, modifier):
    """
    blender for 'ু', 'ূ', 'ৃ'
    """
    # Taking single channel from input images
    alpha = modifier[:, :, 2]  # Shape 14, 14
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]

    # Reshaping input images
    alpha_reshaped = np.hstack((np.zeros((14, 14), np.int8), alpha))  # shape: 14x28
    if letter.shape[0] > 28:
        letter_reshaped = letter
    else:
        letter_reshaped = np.vstack((np.zeros((14, 28), np.int8), letter))  # shape: 42x28
    result = np.vstack((letter_reshaped, alpha_reshaped))
    return result


def blend_modifier_after_top(letter, modifier):
    """
    blender for 'ী'

    """
    # Taking single channel from input images
    alpha = modifier[:, :, 2]  # shape: 42x42
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]

    # Calculating required height for symbol and modifier
    alpha_height_required = 56 - alpha.shape[0]
    letter_height_required = 56 - letter.shape[0]
    if letter_height_required % 2 == 0:
        letter_top = letter_bottom = int(letter_height_required / 2)
    else:
        letter_top = math.floor(letter_height_required / 2) + 1
        letter_bottom = math.floor(letter_height_required / 2)

    # Reshape input images
    alpha_reshaped = np.vstack((alpha, np.zeros((alpha_height_required, alpha.shape[1]), np.int8)))  # shape: 56x42

    if letter.shape[0] > 28:
        letter_reshaped = np.vstack((letter, np.zeros((letter_height_required, letter.shape[1]), np.int8)))
    else:
        letter_reshaped = np.vstack((np.zeros((letter_top, letter.shape[1]), np.int8),
                                     letter, np.zeros((letter_bottom, letter.shape[1]), np.int8)))  # shape: 56x28
    letter_reshaped = np.hstack((letter_reshaped,
                                 np.zeros((letter_reshaped.shape[0], 14), np.int8)))  # shape: 56x42
    result = letter_reshaped + alpha_reshaped
    return result


def blend_modifier_top(letter, modifier, modifier_symbol):
    """
    blender for 'ঁ', 'REF'

    """
    # Taking one channel from input images
    alpha = modifier[:, :, 2]
    if len(letter.shape) > 2:
        letter = letter[:, :, 2]
    # Calculating required height for letter and modifier
    alpha_height_required = 56 - alpha.shape[0]
    letter_height_required = 56 - letter.shape[0]
    # Calculating required width for alpha
    alpha_width_required = letter.shape[1] - alpha.shape[1]
    if alpha_width_required % 2 == 0:
        left = right = int(alpha_width_required / 2)
    else:
        left = math.floor(alpha_width_required / 2) + 1
        right = math.floor(alpha_width_required / 2)

    if modifier_symbol == 'ঁ':
        alpha_reshaped = np.hstack((np.zeros((alpha.shape[0], left), np.int8),
                                    alpha, np.zeros((alpha.shape[0], right), np.int8)))
    else:
        alpha_reshaped = np.hstack((np.zeros((alpha.shape[0], alpha_width_required), np.int8),
                                    alpha))

    # Reshaping letter and modifier where letter height is 42
    if letter.shape[0] > 28:
        alpha_reshaped = np.vstack((alpha_reshaped,
                                    np.zeros((alpha_height_required, alpha_reshaped.shape[1]), np.int8)))
        # letter_reshaped = np.hstack((np.zeros((letter.shape[0], 14), np.int8),
        #                              letter, np.zeros((letter.shape[0], 14), np.int8)))
        letter_reshaped = np.vstack((letter, np.zeros((letter_height_required,
                                                       letter.shape[1]), np.int8)))
        result = letter_reshaped + alpha_reshaped

    # Reshaping letter where letter height is 28
    else:
        # letter_reshaped = np.hstack((np.zeros((28, 14), np.int8), letter, np.zeros((28, 14), np.int8)))
        result = np.vstack((alpha_reshaped, letter, np.zeros((14, letter.shape[1]), np.int8)))
    return result


def get_symbol_image(symbol):
    bangla_character_dict = make_character_to_folder_map_dict(
        parameters.bangla_character_list_file)

    img_folder = bangla_character_dict[symbol]
    img_path = parameters.bangla_character_images_path
    symbol_path = img_path + str(img_folder)
    img_files = os.listdir(symbol_path)
    selected_image_file = random.choice(img_files)

    symbol_img = symbol_path + '/' + selected_image_file
    img_original = cv2.imread(symbol_img, cv2.IMREAD_UNCHANGED)

    return img_original


def get_second_or_third_modifier_blended_image(modified_image, second_modifier, previous_modifier):
    second_modifier_image = get_symbol_image(second_modifier)
    alpha = second_modifier_image[:, :, 2]
    result_stack_flag = False

    # Calculating height and width requirement for second modifier
    alpha_required_width = abs(modified_image.shape[1] - second_modifier_image.shape[1])
    alpha_required_height = modified_image.shape[0] - second_modifier_image.shape[0]

    if alpha_required_width % 2 == 0:
        left = right = int(alpha_required_width / 2)
    else:
        left = math.floor(alpha_required_width / 2) + 1
        right = math.floor(alpha_required_width / 2)

    if alpha_required_height % 2 == 0:
        top = bottom = int(alpha_required_height / 2)
    else:
        top = math.floor(alpha_required_height / 2) + 1
        bottom = math.floor(alpha_required_height / 2)

    # Reshaping second modifier when 'া' and 'JOFOLA'
    if second_modifier == 'া' or second_modifier == 'JOFOLA':
        alpha_reshaped = np.vstack((np.zeros((top, alpha.shape[1]), np.int8), alpha,
                                    np.zeros((bottom, alpha.shape[1]), np.int8)))
        result_stack_flag = True

    # reshaping second modifier when 'ি' and 'ৈ'
    elif second_modifier == 'ি' or second_modifier == 'ৈ':
        # Pad alpha horizontally when alpha has smaller width
        if alpha.shape[1] <= modified_image.shape[1]:
            alpha_reshaped = np.hstack((alpha, np.zeros((alpha.shape[0], alpha_required_width), np.int8)))

        # Pad modified image horizontally when it has smaller width
        else:
            modified_image = np.hstack((np.zeros((modified_image.shape[0], alpha_required_width), np.int8),
                                        modified_image))
            alpha_reshaped = alpha

        # Add extra padding to alpha and modified image if the previous modifier is jofola
        if previous_modifier == 'JOFOLA':
            alpha_reshaped = np.hstack((alpha_reshaped, np.zeros((alpha_reshaped.shape[0], 10))))
            modified_image = np.hstack((np.zeros((modified_image.shape[0], 10), np.int8), modified_image))

        alpha_reshaped = np.vstack((alpha_reshaped,
                                    np.zeros((alpha_required_height, alpha_reshaped.shape[1]), np.int8)))

    # Reshaping second modifier when 'ী'
    elif second_modifier == 'ী':
        if previous_modifier == 'JOFOLA':
            # Pad image horizontally for extra width at the end
            modified_image = np.hstack((modified_image, np.zeros((modified_image.shape[0], 10), np.int8)))

            # Pad alpha horizontally at the beginning
            alpha_reshaped = np.hstack((np.zeros((alpha.shape[0], alpha_required_width), np.int8),
                                        np.zeros((alpha.shape[0], 10), np.int8), alpha))
            # pad alpha vertically
            alpha_reshaped = np.vstack((alpha_reshaped,
                                        np.zeros((alpha_required_height, alpha_reshaped.shape[1]), np.int8)))
        else:
            # Pad alpha horizontally when alpha has smaller width
            if alpha.shape[1] <= modified_image.shape[1]:
                alpha_reshaped = np.hstack((np.zeros((alpha.shape[0], alpha_required_width), np.int8), alpha))

            # Pad modified image horizontally when it has smaller width
            else:
                modified_image = np.hstack((modified_image,
                                            np.zeros((modified_image.shape[0], alpha_required_width), np.int8)))
                alpha_reshaped = alpha
            alpha_reshaped = np.vstack((alpha_reshaped,
                                        np.zeros((alpha_required_height, alpha_reshaped.shape[1]), np.int8)))

    # Reshaping second modifier when 'ে'
    elif second_modifier == 'ে':
        # Pad alpha horizontally when alpha has smaller width
        if alpha.shape[1] <= modified_image.shape[1]:
            alpha_reshaped = np.hstack((alpha, np.zeros((alpha.shape[0], alpha_required_width), np.int8),
                                        np.zeros((alpha.shape[0], 10), np.int8)))

        # Pad modified image horizontally when it has smaller width
        else:
            modified_image = np.hstack((np.zeros((modified_image.shape[0], alpha_required_width), np.int8),
                                        modified_image))
            alpha_reshaped = alpha

        # Pad the image horizontally at the beginning to avoid overlapping with the letter
        modified_image = np.hstack((np.zeros((modified_image.shape[0], 10), np.int8), modified_image))
        alpha_reshaped = np.vstack((np.zeros((top, alpha_reshaped.shape[1]), np.int8), alpha_reshaped,
                                    np.zeros((bottom, alpha_reshaped.shape[1]), np.int8)))

    # Reshaping second modifier when 'ো'
    elif second_modifier == 'ো':
        # Pad modified image horizontally
        modified_image = np.hstack((np.zeros((modified_image.shape[0], left), np.int8),
                                    modified_image, np.zeros((modified_image.shape[0], right), np.int8)))
        alpha_reshaped = np.vstack((np.zeros((top, alpha.shape[1]), np.int8), alpha,
                                    np.zeros((bottom, alpha.shape[1]), np.int8)))
        if previous_modifier == 'JOFOLA':
            split1, split2 = np.hsplit(alpha_reshaped, 2)
            alpha_reshaped = np.hstack((split1, np.zeros((split1.shape[0], 10), np.int8), split2))
            modified_image = np.hstack((np.zeros((modified_image.shape[0], 10), np.int8), modified_image))

    # Reshaping second modifier when 'ৌ'
    elif second_modifier == 'ৌ':
        # Pad modified image horizontally when it has smaller width
        modified_image = np.hstack((np.zeros((modified_image.shape[0], left), np.int8),
                                    modified_image, np.zeros((modified_image.shape[0], right), np.int8)))
        alpha_reshaped = np.vstack((alpha,
                                    np.zeros((alpha_required_height, alpha.shape[1]), np.int8)))

        # Make new alpha image image previous modifier is jofola and pad image accordingly
        if previous_modifier == 'JOFOLA':
            alpha_1 = get_symbol_image('ে')
            alpha_2 = get_symbol_image('OU')
            alpha_1 = alpha_1[:, :, 2]
            alpha_2 = alpha_2[:, :, 2]
            alpha_1_reshaped = np.vstack((np.zeros((14, alpha_1.shape[1]), np.int8),
                                          alpha_1, np.zeros((14, alpha_1.shape[1]), np.int8)))
            alpha_2_reshaped = np.vstack((alpha_2, np.zeros((14, alpha_2.shape[1]), np.int8)))

            alpha_reshaped = np.hstack((alpha_1_reshaped,
                                        np.zeros((alpha_1_reshaped.shape[0], 3), np.int8), alpha_2_reshaped))
            modified_image = np.hstack((np.zeros((modified_image.shape[0], 2), np.int8),
                                        modified_image, np.zeros((modified_image.shape[0], 1), np.int8)))

    # Reshaping second modifier when 'ঁ' and 'REF'
    elif second_modifier == 'ঁ':
        alpha_reshaped = np.hstack((np.zeros((14, left), np.int8), alpha, np.zeros((14, right), np.int8)))
        alpha_reshaped = np.vstack((alpha_reshaped, np.zeros((alpha_required_height,
                                                              alpha_reshaped.shape[1]), np.int8)))
    elif second_modifier == 'REF':
        alpha_reshaped = np.hstack((np.zeros((10, left), np.int8), alpha, np.zeros((18, right), np.int8)))
        alpha_reshaped = np.vstack((alpha_reshaped, np.zeros((alpha_required_height,
                                                              alpha_reshaped.shape[1]), np.int8)))
    else:
        alpha_reshaped = np.hstack((np.zeros((alpha.shape[0], left), np.int8),
                                    alpha, np.zeros((alpha.shape[0], right))))
        alpha_reshaped = np.vstack((np.zeros((alpha_required_height,
                                              alpha_reshaped.shape[1]), np.int8), alpha_reshaped))

    if result_stack_flag:
        result = np.hstack((modified_image, alpha_reshaped))
    else:
        result = modified_image + alpha_reshaped
    return result


def get_modified_image(symbol, modifier):
    if modifier == 'া' or modifier == 'JOFOLA':
        return blend_modifier_after(get_symbol_image(symbol), get_symbol_image(modifier))
    elif modifier == 'ি' or modifier == 'ৈ':
        return blend_modifier_before_top(get_symbol_image(symbol), get_symbol_image(modifier))
    elif modifier == 'ী':
        return blend_modifier_after_top(get_symbol_image(symbol), get_symbol_image(modifier))
    elif modifier == 'ে':
        return blend_modifier_before(get_symbol_image(symbol), get_symbol_image(modifier))
    elif modifier == 'ো':
        return blend_modifier_around(get_symbol_image(symbol), get_symbol_image('ে'), get_symbol_image('া'))
    elif modifier == 'ৌ':
        return blend_modifier_around_top(get_symbol_image(symbol), get_symbol_image(modifier))
    elif modifier == 'ঁ' or modifier == 'REF':
        return blend_modifier_top(get_symbol_image(symbol), get_symbol_image(modifier), modifier)
    else:
        return blend_modifier_bottom(get_symbol_image(symbol), get_symbol_image(modifier))


def seq_to_img(seq, image_name=None, image_path=None):
    img = np.zeros((56, 1), np.int8)
    seq = word_to_symbols(seq)

    i = 0
    seq_len = len(seq)
    modifiers = ['া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', 'ঁ', 'JOFOLA', 'REF']
    while i < seq_len:
        if i <= seq_len - 2 and seq[i + 1] in modifiers:
            symbol = seq[i]
            modifier = seq[i + 1]
            modified_image = get_modified_image(symbol, modifier)
            if i <= seq_len - 3 and seq[i + 2] in modifiers:
                second_modifier = seq[i+2]
                modified_image = get_second_or_third_modifier_blended_image(modified_image, second_modifier, modifier)
                if i <= seq_len - 4 and seq[i + 3] in modifiers:
                    third_modifier = seq[i+3]
                    modified_image = get_second_or_third_modifier_blended_image(modified_image, third_modifier,
                                                                                second_modifier)
                    i += 4
                else:
                    i += 3
            else:
                i += 2
            img = np.hstack((img, modified_image))

        else:

            symbol_image = get_symbol_image(seq[i])
            if len(symbol_image.shape) > 2:
                symbol_image = symbol_image[:, :, 2]
            if symbol_image.shape[0] > 28:
                symbol_image = np.vstack((symbol_image, np.zeros((14, 28), np.int8)))

            else:
                symbol_image = np.vstack((np.zeros((14, 28), np.int8), symbol_image, np.zeros((14, 28), np.int8)))

            img = np.hstack((img, symbol_image))
            i += 1
    if image_name is not None and image_path is not None:
        cv2.imwrite(image_path + image_name, img)
    else:
        return img


def generate_word_images(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            try:
                seq_to_img(line.replace('\n', ''), '{}.png'.format(i),
                           '../test_data/')
            except:
                print('cannot generate image for {}'.format(line))
                continue
            i = i + 1


def generate_static_test_dataset(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        processed = 0
        generated = 0
        failed = 0
        csv_file = open('../asset/handwritten_test.csv', 'w', encoding='utf-8')
        for word_num, line in enumerate(lines):
            if generated >= 500000:
                break
            else:
                for i in range(0, 10):
                    try:
                        image_name = '{}_{}.png'.format(word_num, i)
                        line = line.replace('\n', '')
                        seq_to_img(line, image_name,
                                   '/media/aulipriya/6d389279-103f-4b77-99c5-e24fd8e753dc/home/bjit/Test_Handwritten/')
                        csv_file.write(image_name + ',' + line + '\n')
                        generated += 1
                        print(f'Generated {generated} images')
                    except KeyError:
                        failed += 1
                        print(f'Failed to generate {failed} images')
                processed += 1
                print(f'Processed {processed} images')


# generate_static_test_dataset('../asset/all_words.txt')
# generate_word_images('../asset/nan_test.txt')

# seq_to_img('অপরাধীদের', 'synthetic_word_6.png', '../test_data/')
#get_modified_image('ঠ', 'ি')

# word_to_symbols('তত্ত্ব')
