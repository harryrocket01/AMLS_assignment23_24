
import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

from sklearn.svm import SVC

import matplotlib.pyplot as plt

#Test

from sklearn.metrics import confusion_matrix


from skimage.feature import hog
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray

import tensorflow as tf

from tensorflow.keras import datasets, layers, models,optimizers
from tensorflow.keras.applications import ResNet50

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torchaudio"])



class ML_Template():

    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

class Binary_Exploration(ML_Template):

    def SVM():
        pass

    def XGBoost():
        pass
    
    def CNN(self,optimizer ="Nadam",epochs = 20):
        
        model = Sequential_Models.ResNet50()

        model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        history = model.fit(self.X_train, self.y_train, epochs=epochs, 
                            validation_data=(self.X_val, self.y_val), verbose=1)


        test_loss, test_acc = model.evaluate(self.X_test,  self.y_test, verbose=2)

        print(optimizer,test_loss,test_acc)
        
        return test_loss, test_acc


class Sequential_Models():

    def Default():
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), 
                                activation='relu', 
                                input_shape=((28, 28, 1))))
        
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), 
                                activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), 
                                activation='relu'))

        model.add(layers.Flatten())
        
        model.add(layers.Dense(64, 
                               activation='relu'))
        
        model.add(layers.Dense(10))

        return model
    def Xception():
        pass
    def VGG():
        pass
    def ResNet():


        model = models.Sequential()

        #Add UpSampling layers
        model.add(layers.UpSampling2D((2,2), input_shape=(28, 28, 1)))
        model.add(layers.UpSampling2D((2,2)))
        model.add(layers.UpSampling2D((2,2)))

        #Load ResNet50
        conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Make the ResNet50 layers non-trainable
        for layer in conv_base.layers:
            layer.trainable = False

        model.add(conv_base)

        #Add flattening and dense layers
        model.add(layers.Flatten())
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(10, activation='softmax'))  # Adjust the number of units based on your specific task

        model.summary()
        return model
    
    def MobileNet():
        pass