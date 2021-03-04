import pandas
import os
import numpy as np
import pathlib
import IPython.display as display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("oHAAAAAA 1")

data_directory = pathlib.WindowsPath("./LEGO brick images/train")
CLASSES = np.array([item.name for item in data_directory.glob('*') if item.name != "LICENSE.txt"])
print(CLASSES)
print("oHAAAAAA 2")

image_generator = ImageDataGenerator(rescale=1./255)
dataset = image_generator.flow_from_directory(directory=str(data_directory),
                                                     batch_size=32,
                                                     shuffle=True,
                                                     target_size=(300, 500),
                                                     classes = list(CLASSES))

print("oHAAAAAA 3")
image_batch, label_batch = next(dataset)
print("oHAAAAAA 4")
list_dataset = tf.data.Dataset.list_files(str(data_directory/'*/*'))
print("oHAAAAAA 5")