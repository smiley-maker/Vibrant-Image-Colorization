#Imports

# 
from src.main.lib.dataHandlers.pokemonHandler import pokemonHandler

#Network
from keras.layers import Conv2D, Input, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import numpy as np

#Utilities
import os
import matplotlib.pyplot as plt
import cv2
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

class CNN:
    HUE_WEIGHT = 0.5
    MSE_WEIGHT = 0.1
    SAVE_FILE_PATH = "reports/models/"
    SAVE_MODEL_PATH = "src/saved_models/"
    SHAPE = (256, 256)

    def __init__(self, lr):
        self.lr = lr
        self.input_shape = CNN.SHAPE
        self.input = Input(shape=self.input_shape)
        print(self.input)
        self.output = self.initialize_model(self.input)
        print(self.output)

        self.model = Model(self.input, self.output, name="cnn_256")
        self.model.compile(optimizer=Adam(learning_rate=lr), loss=self.hue_bin_loss)
        
    
    def hue_bin_loss(self, y_true, y_pred):
        print(f"y true has size: {y_true.shape}")
        a_true, b_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
        a_pred, b_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        
        print("a_true shape:", a_true.shape)
        print("b_true shape:", b_true.shape)
        print("a_pred shape:", a_pred.shape)
        print("b_pred shape:", b_pred.shape)

        condition1 = tf.logical_and(tf.less(y_true, 0), tf.less(y_pred, 0))
        condition2 = tf.logical_and(tf.greater(y_true, 0), tf.greater(y_pred, 0))

        hl = self.HUE_WEIGHT * tf.where(condition1 | condition2, 0.0, tf.abs(y_pred - y_true))

        saturation_true = tf.sqrt(tf.square(a_true) + tf.square(b_true))
        saturation_pred = tf.sqrt(tf.square(a_pred) + tf.square(b_pred))
        sl = tf.abs(saturation_true - saturation_pred)

        color_loss = self.MSE_WEIGHT * tf.sqrt(tf.square(a_true - a_pred) + tf.square(b_true - b_pred))
        total_loss = tf.add(color_loss, sl+hl)

        return tf.reduce_mean(total_loss)  # Use reduce_mean to ensure a scalar loss value
    
    
    def train_with_generator(self, handler: pokemonHandler, epochs: int, folder: str, callbacks=[], lr: float = 0.001) -> None:
        if not os.path.exists(CNN.SAVE_FILE_PATH + folder):
            os.makedirs(CNN.SAVE_FILE_PATH + folder)
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'viz'))
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'weights'))
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'images'))

        optimizer = Adam(lr)

        num_steps = len(handler.train_generator)

        print(f"Epochs {epochs}, steps per epoch {num_steps}")

        for epoch in range(epochs):
            for step in range(num_steps):
                X_batch, y_batch = handler.preprocessBatch(next(handler.train_generator))
                print(f"x batch has a shape of: {X_batch.shape}")
                print(f"y batch has a shape of: {y_batch.shape}")

                with tf.GradientTape() as tape:
                    colorized = self.model(X_batch)
                    print(f"colorized has a shape of {colorized.shape}")
                    hue_loss = self.hue_bin_loss(y_batch, colorized)

                    grads = tape.gradient(hue_loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                print(f"epoch {epoch + 1}/{epochs} ::: step {step + 1}/{num_steps} ::: loss {tf.reduce_mean(hue_loss)}")

                # Optionally, you can call your callbacks here
                for func in callbacks:
                    func()

                prob = random.random()
                if prob < 0.1:
                    self.HUE_WEIGHT -= 0.05


    def train(self, x_train, y_train, batchSize : int, epochs : int, folder : str, callBacks = [], lr: float = 0.001) -> None :
        if not os.path.exists(CNN.SAVE_FILE_PATH + folder):            
            os.makedirs(CNN.SAVE_FILE_PATH + folder)
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'viz'))
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'weights'))
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'images'))
            
        optimizer = Adam(lr)

        losses = []
        num_steps = x_train.shape[0]//batchSize

        print(f"Epochs {epochs}, steps per epoch {num_steps}")

        for epoch in range(epochs):
            for step in range(num_steps):
                start_idx = batchSize * step
                end_idx = batchSize * (step + 1)
                indices = np.arange(len(x_train))
                np.random.shuffle(indices)
                selected_indices = indices[start_idx:end_idx]
                X_batch = x_train[selected_indices]
                y_batch = y_train[selected_indices]

                with tf.GradientTape() as tape:
                    colorized = self.model(X_batch)
                    
                    hue_loss = self.hue_bin_loss(y_batch, colorized)
                    losses.append(hue_loss.numpy())                    
                                       
                    grads = tape.gradient(hue_loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
                print(f"epoch {epoch}/{epochs} ::: step {step}/{num_steps} ::: loss {tf.math.reduce_mean(hue_loss)}")

                # for func in callBacks:
                #     func()

                prob = random.random()
                if prob < 0.1:
                    self.HUE_WEIGHT -= 0.05
        
        print(f"Model finished training......")
    
    def initialize_model(self, inp):
        inp_expanded = tf.expand_dims(inp, axis=0) 
        my_model = Conv2D(16, (3, 3), padding='same', strides=1)(inp_expanded)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(32,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(64,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(128,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(256,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(512,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(256,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(128,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(64,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(32,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(16,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)

        my_model = Conv2D(64,(3,3),padding='same',strides=1)(my_model)
        my_model = LeakyReLU()(my_model)
        my_model = BatchNormalization()(my_model)


        my_model = Conv2D(2,(3,3), activation='tanh',padding='same',strides=1)(my_model)

        return my_model
    
    def display_trio(self, img, img_colorized):
        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_original)
        plt.title("Original Image")
        plt.axis('off')

        # Colorized Image
        plt.subplot(1, 3, 2)
        plt.imshow(img_colorized)
        plt.title("Colorized Image")
        plt.axis('off')

        # Ground Truth Image
        plt.subplot(1, 3, 3)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        plt.imshow(img_gray)
        plt.title("Black & White Image")
        plt.axis('off')

        plt.show()
        plt.close()


    def test_model(self, path):
        # Load the test image
        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_ = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        img_=img.astype(np.float32)
        img_lab_rs = cv2.resize(img_, (256, 256)) # resize image to network input size
        img_l = img_lab_rs[:,:,0] # pull out L channel
        img_l_reshaped = img_l.reshape(1,256,256,1)

        # Make a prediction using the trained model
        Prediction = self.model.predict(img_l_reshaped)
        Prediction = Prediction * 128
        Prediction = Prediction.reshape(256, 256, 2)

        # Create Lab image with L channel from original image and predicted a,b channels
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        img_lab_colorized = img_lab.copy()
        img_lab_colorized[:, :, 1:] = Prediction

        # Convert Lab image to RGB
        img_colorized = cv2.cvtColor(img_lab_colorized, cv2.COLOR_Lab2RGB)

        # Display the images
        self.display_trio(img, img_colorized)

        # Calculate loss
        loss = self.hue_bin_loss(img_lab_colorized[:, :, 1:], Prediction)
        print(f"test image loss: {loss}")
    

    def save_model(self, version : str = "v0"):
        self.model.save(f"{CNN.SAVE_MODEL_PATH}CNN--{version}.h5")
        print(f"Model saved as {CNN.SAVE_MODEL_PATH}CNN--{version}.h5 in saved_models.")
    
    def load_model(self, version : str = "v0"):
        self.model = load_model(f"{CNN.SAVE_MODEL_PATH}CNN--{version}.h5", custom_objects={"loss": self.hue_bin_loss}) 

    def save(self, folder : str) -> None :
        if not os.path.exists(CNN.SAVE_FILE_PATH + folder):
            os.makedirs(CNN.SAVE_FILE_PATH + folder)
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'viz'))
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'weights'))
            os.makedirs(os.path.join(CNN.SAVE_FILE_PATH + folder, 'images'))

        with open(os.path.join(CNN.SAVE_FILE_PATH + folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_shape,
                self.epochs,
                self.lr
                ], f)