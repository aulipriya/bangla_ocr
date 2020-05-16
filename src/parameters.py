max_image_height = 56
max_image_width = 512

number_of_classes = 73
train_text_file_path = '../../data/printed.csv'
validation_text_file_path = '../../data/printed.csv'
test_text_file_path='../asset/handwritten_test.csv'
train_data_dir='../data/train/'
val_data_dir='../data/val/'
test_data_dir='../data/test/'


# Data generation parameters

bangla_character_list_file = '../../asset/bangla_bornomala2.txt'
bangla_character_images_path = '../../data/banglalekha_resized_2/'
bangla_words_list = '../../asset/test_label.txt'
font_files_root_path = '../../asset/fonts/'
font_names_list_path = '../../asset/fonts/font_names.txt'
generated_images_save_dir = '../../data/printed'
generated_csv_path = '../../data/printed.csv'
generation_method = 'printed'


# Training parameters
train_root_directory = '../../data/printed/'
val_root_directory = '../../data/printed/'
epochs = 3
input_channels = 3
batch_size = 5
learning_rate = 0.0001
training_details_file = '../../asset/mobilenet_training_details.csv'
validation_details_file = '../../asset/mobilenet_validation_details.csv'
weights_folder = '../../weights/'
display_interval = 100
validation_interval = 1

