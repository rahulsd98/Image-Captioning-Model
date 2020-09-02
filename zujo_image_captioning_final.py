# -*- coding: utf-8 -*-
"""Zujo Image Captioning Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NRNJ33aEaRR3nl_P81nZ4Hr5WxYl1mcS
"""

#importing pandas library for reading the dataset
import pandas as pd
import seaborn as sns

from google.colab import files

files.upload()

!ls -lha kaggle.json

!pip install -q kaggle

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

#Downloading the dataset from kaggle
!kaggle datasets download -d paramaggarwal/fashion-product-images-small

!ls

!unzip /content/fashion-product-images-small.zip

#Reading the dataset using pandas
df = pd.read_csv('/content/myntradataset/styles.csv', error_bad_lines = False)

df.dtypes

df.isnull().mean() * 100

df = df.dropna()

df.isnull().mean() * 100

df.head()

cat = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName']
for i in cat:
  df[i] = df[i].astype('category')

df.dtypes

df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis = 1)
df = df.reset_index(drop=True)

df.head()

df.masterCategory.unique()

df.shape

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
code = np.array(df['masterCategory'])
label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(code)
target = to_categorical(vec)

# Commented out IPython magic to ensure Python compatibility.
import sklearn
import math
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# %matplotlib inline

batch_size = 256

image_generator = ImageDataGenerator(validation_split=0.3)

training_generator = image_generator.flow_from_dataframe(
    dataframe = df,
    directory = "images",
    x_col = "image",
    y_col = "masterCategory",
    target_size = (80,60),
    batch_size = batch_size,
    subset = "training"
)

validation_generator = image_generator.flow_from_dataframe(
    dataframe = df,
    directory = "images",
    x_col = "image",
    y_col = "masterCategory",
    target_size = (80,60),
    batch_size = batch_size,
    subset = "validation"
)
classes = len(training_generator.class_indices)

from keras.utils import np_utils

model = Sequential()

input_shape = (1,80,60,3)

#Add the necessary layers
model.add(layers.Conv2D(32, kernel_size=(5,5), strides=(2,2), activation='relu', input_shape = input_shape[1:]))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))

model.add(layers.Dense(7,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit_generator(
    generator = training_generator,
    steps_per_epoch = math.ceil(0.7 * (df.shape[0] / batch_size)),
    validation_data = validation_generator,
    validation_steps = math.ceil(0.3 * (df.shape[0] / batch_size)),
    epochs = 5, verbose = 1
)

steps = math.ceil(0.3 * (df.shape[0]/batch_size))
score = model.evaluate(validation_data, steps = steps)
print("Test Loss: ",score[0]*100)
print("Test Accuracy: ",score[1]*100)

import h5py
model.save('fashion_image_captioning_model.h5')