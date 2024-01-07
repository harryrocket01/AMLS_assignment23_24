"""
AMLS 1 Final Assessment - MV_NN.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

#Custom Function Imports
from Models.Template import ML_Template

#Python Package Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras import datasets, layers, models,optimizers
from tensorflow.keras.applications import ResNet50


class MV_CNN(ML_Template):

    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)
        self.name = ""
        self.model = None

        self.history = None

        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.001


    def SetModel(self,Model = "NN"):
        Model = Model.lower()
        if Model == "resnet":
            self.model = Sequential_Models.ResNet()
        elif Model == "resnet2":
            self.model = Sequential_Models.ResNet_2()
        elif Model == "resnet50":
            self.model = Sequential_Models.ResNet_50()
        elif Model == "taska":
            self.model = Sequential_Models.CNN_A()
        elif Model == "deep":
            self.model = Sequential_Models.Deep()
        else:
            self.model = Sequential_Models.Default()
            print(self.model.summary())
        
        self.name = Model



    #based off of the Keras Documentation

    def Train(self,verbose:int = 1):

        #Check if there is a file
        filename = 'B\Models\PreTrainedModels\{}Model.keras'.format(self.name)

        try:
            self.model = tf.keras.models.load_model(filename)
            if verbose:
                print("\nPre Trained Model Loaded - {}Model.keras".format(self.name))

            return None
        except:
            if verbose:
                print("No Pre Trained Model, Training Model - {}Model.keras\n".format(self.name))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            history = self.TrainLoop()
            self.model.save(filename)
            
            return history


    def SetHyperPerameters(self, epochs:int =None,batch_size: int = None,learning_rate: float = None):
        self.epochs = epochs if epochs else self.epochs
        self.batch_size = batch_size if batch_size else self.batch_size
        self.learning_rate = learning_rate if learning_rate else self.learning_rate

    def TrainLoop(self, verbose: str = 1):

        history = {}

        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss  = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        train_history = self.model.fit(self.X_train, self.y_train,  
                                        epochs=self.epochs, 
                                        batch_size=self.batch_size,
                                        validation_data=(self.X_val, self.y_val),
                                        verbose = verbose)
        
        history["train_acc"] = train_history.history['accuracy']
        history["train_loss"] = train_history.history['loss']
        history["val_loss"] = train_history.history['val_loss']
        history["val_acc"] = train_history.history['val_accuracy']

        return history

    def Custom_TrainLoop(self,verbose: int = 1):

        history = {"train_acc":[],"train_loss":[],"val_loss":[],"val_acc":[],"train_time":[]}


        #Defining optomisor and loss function
        optimizer = keras.optimizers.Adam(self.learning_rate)
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)

        #Converting Traning dataset to tensor
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        #Converting Validation dataset to tensor.
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_dataset = val_dataset.batch(self.batch_size)

        #Defidning accuracy metrics
        train_acc_metric = keras.metrics.CategoricalAccuracy()
        val_acc_metric = keras.metrics.CategoricalAccuracy()


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

           
            history["train_acc"].append(train_acc.numpy())
            history["train_loss"].append(train_loss.numpy())
            history["val_loss"].append(val_loss.numpy())
            history["val_acc"].append(val_acc.numpy())
            history["train_time"].append(train_time)
            
            
            if verbose:
                to_print = "Epoch {}/{} - {:.1f} - loss: {:.4f} - accuracy: {:.2f} - val_loss: {:.4f} - val_accuracy: {:.4f}"
                to_print = to_print.format(epoch+1,self.epochs,
                                        train_time,
                                        train_loss,train_acc,
                                        val_loss,val_acc)
                print(to_print)
        return history                                                                                        
    
    def Test(self, verbose: int = 1) -> (int, ArrayLike):

        test_dataset = tf.data.Dataset.from_tensor_slices((self.X_test, self.y_test))
        test_dataset = test_dataset.batch(self.batch_size) 

        test_acc_metric = tf.keras.metrics.CategoricalAccuracy() 
        y_pred = []

        #Loop over the test set
        for x_batch_test, y_batch_test in test_dataset:
            test_logits = self.model(x_batch_test, training=False)
            batch_pred = (test_logits >= 0.5).numpy().astype(int)
            test_acc_metric.update_state(y_batch_test, test_logits)
            y_pred.extend(batch_pred)
            #y_pred.extend(batch_pred)

        test_acc = test_acc_metric.result()
        if verbose:
            print("Test accuracy: {:.4f}".format(float(test_acc)))

        #Reset the test accuracy metric
        test_acc_metric.reset_states()

        y_pred_np = np.array(y_pred)

        print(y_pred_np.tolist())

        #convert back from one hot encoded
        y_pred_np = np.argmax(y_pred_np,  axis=1)

        return float(test_acc), y_pred_np





class Sequential_Models():

    def Default():
        #Suggested model from Tensor Flow
        model = models.Sequential()
        model.add(layers.Input(shape=(28, 28, 3)))

        model.add(layers.Conv2D(32, (3, 3), 
                                activation='relu'))
        
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), 
                                activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), 
                                activation='relu'))

        model.add(layers.Flatten())
        
        model.add(layers.Dense(64, 
                               activation='relu'))
        
        model.add(layers.Dense(9, activation="softmax"))

        return model
    
    def CNN_A(dropout_rate: float = 0.25):
            model = models.Sequential()

            model.add(layers.Input(shape=(28, 28, 3)))


                
            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu'))
            
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Dropout(dropout_rate))

            model.add(layers.Conv2D(64, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.Conv2D(64, (3, 3), 
                                        activation='relu'))

            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Flatten())
            
            model.add(layers.Dropout(0.5))

            model.add(layers.Dense(64, 
                                activation='relu'))
            
            model.add(layers.Dense(9, activation="softmax"))

            return model

        
    def Deep(dropout_rate: float = 0.25):
            model = models.Sequential()

            model.add(layers.Input(shape=(28, 28, 3)))


            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))

            model.add(layers.GlobalAveragePooling2D())
            
            model.add(layers.Dropout(0.5))

            model.add(layers.Dense(64, 
                                activation='relu'))
            
            model.add(layers.Dense(9, activation="softmax"))

            return model

    def ResNet_50():
        model = models.Sequential()

        model.add(layers.Input(shape=(28, 28, 3)))
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
        model.add(layers.Dense(9, activation='softmax'))

        return model

    def ResNet(dropout_rate: int = 0.5):
        inputs = keras.Input(shape=(28, 28, 3))
        main_path = layers.Conv2D(32, (3, 3), 
                                  activation="relu")(inputs)
        main_path = layers.Conv2D(64, (3, 3), 
                                  activation="relu")(main_path)

        output_1 = layers.MaxPooling2D((2,2))(main_path)

        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(output_1)
        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(main_path)
        output_2 = layers.add([main_path, output_1])

        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(output_2)
        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(main_path)
        block_3_output = layers.add([main_path, output_2])
       
        main_path = layers.Conv2D(64, (3, 3), 
                                  activation="relu")(block_3_output)

        main_path = layers.MaxPooling2D((2,2))(main_path)

        main_path = layers.GlobalAveragePooling2D()(main_path)
        main_path = layers.Dense(128, 
                                 activation="relu")(main_path)
        main_path = layers.Dropout(dropout_rate)(main_path)
        outputs = layers.Dense(9, activation="softmax")(main_path)

        model = keras.Model(inputs, outputs, name="Residual_Network")

        model.summary()
        return model

    def ResNet_2(dropout_rate: int = 0.5):
        inputs = keras.Input(shape=(28, 28, 3))
        main_path = layers.Conv2D(32, (3, 3), 
                                  activation="relu")(inputs)
        main_path = layers.Conv2D(64, (3, 3), 
                                  activation="relu")(main_path)

        output_1 = layers.MaxPooling2D((2,2))(main_path)

        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(output_1)
        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(main_path)
        output_2 = layers.add([main_path, output_1])

        main_path = layers.Conv2D(128, 3, activation="relu", 
                                  padding="same")(output_2)
        main_path = layers.Conv2D(128, 3, activation="relu", 
                                  padding="same")(main_path)
        output_3 = layers.add([main_path, output_2])

        main_path = layers.Conv2D(128, 3, activation="relu", 
                                  padding="same")(output_3)
        main_path = layers.Conv2D(128, 3, activation="relu", 
                                  padding="same")(main_path)
        output_4 = layers.add([main_path, output_3])

        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(output_2)
        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(main_path)
        block_5_output = layers.add([main_path, output_4])
       
        main_path = layers.Conv2D(64, (3, 3), 
                                  activation="relu")(block_5_output)

        main_path = layers.MaxPooling2D((2,2))(main_path)

        main_path = layers.GlobalAveragePooling2D()(main_path)
        main_path = layers.Dense(128, 
                                 activation="relu")(main_path)
        main_path = layers.Dropout(dropout_rate)(main_path)
        outputs = layers.Dense(9, activation="softmax")(main_path)

        model = keras.Model(inputs, outputs, name="Residual_Network")

        model.summary()
        return model