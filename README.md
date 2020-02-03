# Bangla OCR
### Description
This project started with an aim to convert bangla handwritten images to text, i.e digitize the handwritten documents. Due to inadequate amount of data for handwritten words we started the experiment with printed word images. After that we moved on to work with  handwritten word images by synthetically generating them.  

### Dataset 
We generated the dataset synthetically, the code of synthetic data generation can be found in the data_preparation directory.

### Model 
This project takes inspiration from [this](https://arxiv.org/abs/1507.05717) paper for the model and [this](https://github.com/meijieru/crnn.pytorch) repo for code.

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
