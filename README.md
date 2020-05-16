# Bangla OCR
### Description
This project started with an aim to convert bangla handwritten images to text, i.e digitize the handwritten documents. Due to inadequate amount of data for handwritten words we started the experiment with printed word images. After that we moved on to work with  handwritten word images by synthetically generating them.  

### Prerequisites 
* opencv 
* pytorch
* pillow

### Install Requirements 
```Language
pip install -r requirements.txt
```
### Run 
```Language
python training/train_mobilenet.py
```


### Dataset 
We generated the dataset synthetically, the code of synthetic data generation can be found in the data_preparation directory.To generate data - 
```Language
python data_preparation/data_generator.py
```
Generated data will be saved in the data/printed_data directory, to change the path of the saved images, change the parameter generated_images_save_dir in parameters.py file. The list of the words to be converted to images are by deafult a txt file saved in asset directory. This can also be changed from the parameters file by bangla_words_list parameter. 
In case of printed data generation, the fonts must be saved in the asset/fonts/ directory, this path can be changed by the parameter font_files_root_path in the parameter file.
Names of these fonts are saved in the asset/fonts/font_names.txt file, this can be changed by the parameter font_names_list_path. 
In case of handwritten data generation, for character images [CMATERDB](https://www.dropbox.com/s/55bhfr3ycvsewsi/CMATERdb%203.1.2.rar) dataset. We also added custom images for modifiers 'a-kar', 'o-kar'so on to build our custum data set. Where most of the characters are resized to 28*28 and the modifiers are resized to various sizes as per need. The file path of this resized character images can be changed by bangla_character_images_path. The mapping of folders for different character is saved in the asset/bangla_bornomala2.txt file. It can be changed by bangla_character_list_file parameter.
### Model 
This project takes inspiration from [this](https://arxiv.org/abs/1507.05717) paper for the model and [this](https://github.com/meijieru/crnn.pytorch) repo for code.
We prepared two models, one from the stated paper, and the other taking inspiration from the paper replaced the CNN part with mobilenet. Although the best results were still yielded by the paper.All the implemented models can be found in the models directory. 
### Results 
Following are some of the results from one of many experiments.


Prediction : ডিপ্লয়েড   GT : ডিপ্লয়েড

Prediction : প্যাজারামপাড়া   GT : প্যাজারামপাড়া

Prediction : বেনকেনের   GT : বেনকেনের

Prediction : মেটাল   GT : মেটাল

Prediction : লুথেনবাগ   GT : লুথেনবাগ

Prediction : বুরিমাড়ী   GT : বুরিমাড়ী

Prediction : লিভোমিসল   GT : লিভোমিসল

Prediction : রনৌত   GT : রনৌত

Prediction : দক্রশিক্ষকের   GT : দক্ষ্রশিক্ষকের

Prediction : পাইভা   GT : পাইভা


We achieved 93.5 % test accuracy for printed data and 94.273 % test accuracy for handwritten data. 
