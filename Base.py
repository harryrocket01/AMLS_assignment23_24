

#
#
#
#
#
#

# ---------- Imports ----------
import subprocess
import sys

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


from skimage.feature import hog
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray

#import tensorflow as tf

#from tensorflow.keras import datasets, layers, models







class Data_Processing():
    def __init__(self) -> None:
        pass


class Final_Binary_Classifcation():

    def __init__(self) -> None:
        pass

    def SVM(self,X_train, X_test,X_val,y_train,y_test,y_val) -> None:
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

        # Train the model
        svm_model.fit(X_train, y_train)

        # Validate the model
        val_predictions = svm_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print("Validation accuracy:", val_accuracy)
        print(classification_report(y_val, val_predictions))

        # Test the model
        test_predictions = svm_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        print("Test accuracy:", test_accuracy)
        print(classification_report(y_test, test_predictions))


    def CNN_Conv2D():
        Model = models.Sequential()
        Model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        Model.add(layers.MaxPooling2D((2, 2)))
        Model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        Model.add(layers.MaxPooling2D((2, 2)))
        Model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        Model.add(layers.Flatten())
        Model.add(layers.Dense(64, activation='relu'))
        Model.add(layers.Dense(10))

        return Model

    def CNN_Train(Model, X_train,y_train,X_val,y_val):
        Model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        Trained_Model = Model.fit(X_train, y_train, epochs=10, validation_data=(X_val,y_val))
        return Trained_Model
    
    def Final_Test(Model,X_test,y_test):
        test_loss, test_acc = Model.evaluate(X_test,  y_test, verbose=2)

        print(test_loss, test_acc)
        return test_loss, test_acc



class Final_Multivariate_Classifcation():

    def __init__(self) -> None:
        pass

    def Train()-> None:
        pass
    
    def Validate()-> None:
        pass

    def Test()-> None:
        pass


class ML_Template():

    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

class Binary_Exploration(ML_Template):
    
    def CNN(self,optimizer ="Nadam",epochs = 20):
        
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=((28, 28, 1))))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))

        model.compile(optimizer=optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        history = model.fit(self.X_train, self.y_train, epochs=epochs, 
                            validation_data=(self.X_val, self.y_val), verbose=0)


        test_loss, test_acc = model.evaluate(self.X_test,  self.y_test, verbose=2)

        print(optimizer,test_loss,test_acc)
        
        return test_loss, test_acc

class Multivariate_Exploration(ML_Template):

    def __init__(self):
        pass





class Plotting ():

    def __init__(self) -> None:
        pass

    def Confusion_Matrix()-> None:
        pass
    
    def Line_Plot(x: ArrayLike, y:ArrayLike, title: str, x_label: str, y_label: str, legend) -> (plt.Figure, plt.Axes):
        fig, ax = plt.subplots()
        counter = 1

        if isinstance(y, np.ndarray) and y.ndim == 2:
            for Y_current in y:
                ax.plot(x, Y_current)
        else:
            ax.plot(x, y)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(legend)




PathMNIST = np.load('Datasets\pathmnist.npz')
PneumoniaMNIST = np.load('Datasets\pneumoniamnist.npz')



print("Keys in PathMNIST: ", list(PathMNIST.keys()))
print("Keys in PneumoniaMNIST: ", list(PneumoniaMNIST.keys()))


#Extract Path Data

#Train
Path_train_images = PathMNIST['train_images']
Path_train_labels = PathMNIST['train_labels']

#Validation
Path_val_images = PathMNIST['val_images']
Path_val_labels = PathMNIST['val_labels']

#Test
Path_test_images = PathMNIST['test_images']
Path_test_labels = PathMNIST['test_labels']


#Extract Pneumonia

#Train
Pneumonia_train_images = PneumoniaMNIST['train_images']
Pneumonia_train_labels = PneumoniaMNIST['train_labels']

#Validation
Pneumonia_val_images = PneumoniaMNIST['val_images']
Pneumonia_val_labels = PneumoniaMNIST['val_labels']

#Test
Pneumonia_test_images = PneumoniaMNIST['test_images']
Pneumonia_test_labels = PneumoniaMNIST['test_labels']


# Flatten the images for scikit-learn compatibility
n_train_samples = len(Pneumonia_train_labels)
n_val_samples = len(Pneumonia_val_labels)
n_test_samples = len(Pneumonia_test_labels)

X_train = Pneumonia_train_images.reshape((n_train_samples, 28, 28, 1))
X_val = Pneumonia_val_images.reshape((n_val_samples, 28, 28, 1))
X_test = Pneumonia_test_images.reshape((n_test_samples, 28, 28, 1))

y_train = Pneumonia_train_labels
y_val = Pneumonia_val_labels
y_test = Pneumonia_test_labels



Temp = Final_Binary_Classifcation()

Temp.SVM(X_train,y_train,X_val,y_val,X_test,y_test)
#Temp.CNN("RMSprop")
#Temp.CNN("Adagrad")
#Temp.CNN("Nadam")
