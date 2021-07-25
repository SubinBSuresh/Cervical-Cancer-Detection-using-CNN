# -*- coding: utf-8 -*-

# Import Libraries

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow import keras
from IPython.display import Image
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import pathlib
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class_names=['Dyskeratotic','Koilocytotic','Metaplastic','parabasal','Superficial-Intermediate']

# Loading the Saved Model

model=load_model('/home/subin/PROJECT/MODELS/custom.h5')

model.summary()

pip install keras_sequential_ascii

from keras_sequential_ascii import keras2ascii
keras2ascii(model)

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Determining the size of the data or dataset

batch_size = 4 #3
img_height = 224 #180
img_width = 224 #180

# Load Test data
test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(
    '/home/subin/PROJECT/DATASETS/Sipakmed/TEST', 
     target_size=(img_width, img_height),
    batch_size=10,
    shuffle=False)

# Acquire actual class of test dataset

true_class = test_data_generator.classes
print(true_class)

# Predicting Testing dataset

test_pred = model.predict(test_data_generator)
print(test_pred)

predicted_class = np.argmax(test_pred,axis=1)
print(predicted_class)

# Classification Report

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(true_class, predicted_class))

# Confusion Matrix

import seaborn as sns
cf_matrix = confusion_matrix(true_class, predicted_class)
sns.heatmap(cf_matrix, annot=True, cmap='Blues')

# Individual Image Prediction

path = "/home/subin/PROJECT/DATASETS/Herlev/TEST/HSIL_jpg/149056321-149056360-003_180.jpg"
img = keras.preprocessing.image.load_img(
path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
display(Image(filename=path))
print(
"This is a {}."
.format(class_names[np.argmax(score)])
)
