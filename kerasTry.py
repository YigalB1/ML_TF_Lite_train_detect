# from https://keras.io/examples/vision/image_classification_from_scratch/

# also from https://www.youtube.com/watch?v=cvyDYdI2nEI&t=985s
# his git with all data: https://github.com/TannerGilbert/Tensorflow-Object-Detection-API-Train-Model

print("starting")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("done importing")

pets_pics_dir = "PetImages"
trainBR50_pics_dir = "TrainBR50"


num_skipped = 0
#for folder_name in ("Cat", "Dog"):
for folder_name in ("TrainBR50"):
    #folder_path = os.path.join(pets_pics_dir, folder_name)
    folder_path = trainBR50_pics_dir
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

print("done dbase")
print("Visualize the data")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

print("The end")
