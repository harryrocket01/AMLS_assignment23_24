
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import time

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras
from keras import layers

from tensorflow.keras import datasets, layers, models,optimizers
from tensorflow.keras.applications import ResNet50

from Models.Template import ML_Template

class NN(ML_Template):

    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)
        self.name = ""
        self.model = None

        self.history = None

        self.epochs = 5
        self.batchsize = 64
        self.learningrate = 0.001


    def SetModel(self,Model = "NN"):
        Model = Model.lower()
        if Model == "resnet":
            self.model = Sequential_Models.ResNet()
        elif Model =="ManyLayers":
            self.model = Sequential_Models.ManyLayers()
        elif Model =="Alt":
            self.model = Sequential_Models.Alt()
        elif Model =="":
            pass
        else:
            self.model = Sequential_Models.Default()
        
        print(self.model.summary())
        self.name = Model



    #based off of the Keras Documentation

    def Train(self,epochs=None,batchsize = None,learningrate = None):

        self.epochs = epochs if epochs else self.epochs
        self.batchsize = batchsize if batchsize else self.batchsize
        self.learningrate = learningrate if learningrate else self.learningrate


        #Check if there is a file
        filename = 'A\Models\PreTrainedModels\{}Model.keras'.format(self.name)
        try:
            self.model = tf.keras.models.load_model(filename)

        except:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.Custom_TrainLoop()
            self.model.save(filename)

    def TrainLoop(self):
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningrate)
        loss  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train,  
                        epochs=self.epochs, 
                        batch_size=self.batchsize,
                        validation_data=(self.X_val, self.y_val))

    def Custom_TrainLoop(self,):

        training_losses = []
        training_accuracies = []
        validation_accuracies = []


        #Defining optomisor and loss function
        optimizer = keras.optimizers.Adam(self.learningrate)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        
        #self.model.compile(optimizer=optimizer,
        #            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
         #           metrics=['accuracy'])
        
        #history = self.model.fit(self.X_train, self.y_train, epochs=epochs, 
                            #validation_data=(self.X_val, self.y_val), verbose=1)
        
        #Converting Traning dataset to tensor
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batchsize)

        #Converting Validation dataset to tensor.
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_dataset = val_dataset.batch(self.batchsize)

        #Defidning accuracy metrics
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


        #Main Training Loop
        for epoch in range(self.epochs):
            start_time = time.time()

            #Batch Steps within traning set
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    train_loss = loss(y_batch_train,logits)
                grads = tape.gradient(train_loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                train_acc_metric.update_state(y_batch_train, logits)

            # Display train metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            # Reset training metrics
            train_acc_metric.reset_states()

            # Runs Epoch Validation .
            for x_batch_val, y_batch_val in val_dataset:
                val_logits = self.model(x_batch_val, training=False)
                # Update val metrics
                val_loss = loss(y_batch_val, val_logits)  # Compute validation loss
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()

            train_time = (time.time() - start_time)

            print("Epoch {}/{} - {:.1f} - loss: {:.4f} - accuracy: {:.2f} - val_loss: {:.4f} - val_accuracy: {:.4f}".format(epoch,self.epochs,
                                                                                                                            train_time,
                                                                                                                            train_loss,train_acc,
                                                                                                                            val_loss,val_acc))


    def Test(self):
        test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        test_dataset = test_dataset.batch(self.batchsize)  # Use the same batch size as in training

        test_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        # Loop over the test set
        for x_batch_test, y_batch_test in test_dataset:
            test_logits = self.model(x_batch_test, training=False)  # Set training=False for testing
            test_acc_metric.update_state(y_batch_test, test_logits)

        test_acc = test_acc_metric.result()
        print("Test accuracy: %.4f" % (float(test_acc),))

        # Reset the test accuracy metric
        test_acc_metric.reset_states()

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
        
        model.add(layers.Dense(2))

        return model
    
    def ManyLayers():
        model = models.Sequential()
        model.add(layers.Input(shape=(28, 28, 3)))

        Filters = [32,64,128,128,64,64]
        for current in Filters:
            model.add(layers.Conv2D(current, (3, 3), 
                                    activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), 
                                activation='relu'))

        model.add(layers.Flatten())
        
        model.add(layers.Dense(16, 
                               activation='relu'))
        
        model.add(layers.Dense(2))
        return model

    
    def Alt():
        model = models.Sequential()


        Filters = [32,64,128,32,128,32,128,64]
        for current in Filters:
            model.add(layers.Conv2D(current, (3, 3), 
                                    activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        
        model.add(layers.Dense(64, 
                               activation='relu'))
        
        model.add(layers.Dense(1))
        return model
    
    def Uber():
        model = models.Sequential()


        Filters = [32, 64, 128, 256,256,256, 128, 64]
        for current in Filters:
            model.add(layers.Conv2D(current, (3, 3), 
                                    activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        
        model.add(layers.Dense(64, 
                               activation='relu'))
        
        model.add(layers.Dense(1))
        return model


    def ResNet():
        model = models.Sequential()

        model.add(layers.Input(shape=(28, 28, 1)))
        model.add(layers.experimental.preprocessing.Resizing(56, 56))
        model.add(layers.Conv2D(3, (1, 1), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        # ResNet50
        resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(56, 56, 3))
        model.add(resnet_model)
        model.add(layers.GlobalAveragePooling2D())

        # Additional custom layers
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2, activation='softmax'))

        return model
    
    def MobileNet():
        pass