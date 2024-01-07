# Sets up places dataset and preprocessing. 

#import tensorflow as tf
#import tensorflow_datasets as tfds

import cv2
import os
import numpy as np

# Load the dataset
#print("Loading the dataset...")
#(ds_train, ds_test) = tfds.load('imagenette', split='train', shuffle_files=True, data_dir="../data/places/")

#print("Loaded")

# Split the dataset into train, validation, and test
#train_ds = ds['train']
#val_ds = ds['validation']
#test_ds = ds['test']

# Print some information about the dataset
#print(tfds.as_dataframe(ds_train.take(4)))


class imagenetteHandler:
    BASE_PATH = "src/data/places/downloads/extracted/data/imagenette2/"

    def __init__(self, image_size: tuple = (256, 256)):
        self.image_size = image_size

        print("Gathering data: ")
        self.x_train, self.y_train = self.prepareInputData(imagenetteHandler.BASE_PATH + "train/train_data/")
        self.x_test, self.y_test = self.prepareInputData(imagenetteHandler.BASE_PATH + "val/val_data/")
        print("x_train shape:", self.x_train.shape)
        print("y_train shape:", self.y_train.shape)


    def prepareInputData(self, path):
        X = []
        y = []

        for imageDir in os.listdir(path):
            try:
                img = cv2.imread(path + imageDir)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                img = img.astype(np.float32)
                img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
                img_lab_rs = cv2.resize(img_lab, self.image_size)  # resize image to network input size
                # Resize both the L channel and ab channels to the network input size
                img_l = img_lab_rs[:, :, 0]
                img_ab = img_lab_rs[:, :, 1:] / 128.0

                X.append(img_l)
                y.append(img_ab)
            except:
                pass

        X = np.array(X)
        y = np.array(y)

        return X, y
