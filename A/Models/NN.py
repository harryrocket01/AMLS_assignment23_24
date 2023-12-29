
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


    def SetModel(self,model = "NN",verbose: int = 1,dropout_rate: float = 0.5, layer: ArrayLike = [32,64,64]) -> None:
        model = model.lower()
        if model == "resnet":
            self.model = Sequential_Models.ResNet(dropout_rate = dropout_rate)
        elif model =="ManyLayers":
            self.model = Sequential_Models.CNN(dropout_rate=dropout_rate, layer=[32, 64, 128, 256,256,256, 128, 64])
        elif model =="Alt":
            self.model = Sequential_Models.CNN(dropout_rate=dropout_rate, layer= [32,64,128,32,128,32,128,64])
        elif model =="":
            pass
        else:
            self.model = Sequential_Models.CNN(dropout_rate=dropout_rate, layer=layer)
        
        self.name = model

        if verbose:
            print(self.model.summary())




    #based off of the Keras Documentation

    def Train(self,epochs=None,batchsize = None,learningrate = None,verbose: int = 1) -> dict:

        self.epochs = epochs if epochs else self.epochs
        self.batchsize = batchsize if batchsize else self.batchsize
        self.learningrate = learningrate if learningrate else self.learningrate


        #Check if there is a file
        filename = 'A\Models\PreTrainedModels\{}Model.keras'.format(self.name)
        try:
            self.model = tf.keras.models.load_model(filename)
            
            if verbose:
                print("\nPre Trained Model Loaded - {}Model.keras".format(self.name))
            return None
        except:
            if verbose:
                print("No Pre Trained Model, Training Model - {}Model.keras\n".format(self.name))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            history = self.Custom_TrainLoop()
            self.model.save(filename)
            return history

    def train_one_off(self,epochs=None,batchsize = None,learningrate = None,verbose: int = 0):
        
        self.epochs = epochs if epochs else self.epochs
        self.batchsize = batchsize if batchsize else self.batchsize
        self.learningrate = learningrate if learningrate else self.learningrate
        history = self.TrainLoop(verbose=verbose)

        return history["train_acc"][-1], history["train_loss"][-1], history["val_acc"][-1], history["val_loss"][-1]



    def TrainLoop(self,verbose: int = 1) -> dict:

        history = {}

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningrate)
        loss = tf.keras.losses.BinaryCrossentropy()

        self.model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
        train_history = self.model.fit(self.X_train, self.y_train,  
                                        epochs=self.epochs, 
                                        batch_size=self.batchsize,
                                        validation_data=(self.X_val, self.y_val),
                                        verbose = verbose)
        
        history["train_acc"] = train_history.history['accuracy']
        history["train_loss"] = train_history.history['loss']
        history["val_loss"] = train_history.history['val_loss']
        history["val_acc"] = train_history.history['val_accuracy']

        return history

    def Custom_TrainLoop(self,verbose: int = 1) -> dict:

        history = {"train_acc":[],"train_loss":[],"val_loss":[],"val_acc":[],"train_time":[]}


        #Defining optomisor and loss function
        optimizer = keras.optimizers.Adam(self.learningrate)


        
        loss = tf.keras.losses.BinaryCrossentropy()

        #Converting Traning dataset to tensor
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batchsize)

        #Converting Validation dataset to tensor.
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val_dataset = val_dataset.batch(self.batchsize)

        #Defidning accuracy metrics
        train_acc_metric = keras.metrics.BinaryAccuracy() 
        val_acc_metric = keras.metrics.BinaryAccuracy()


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
        test_dataset = test_dataset.batch(self.batchsize) 

        test_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5) 
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
            print("Test accuracy: %.4f" % (float(test_acc)))

        #Reset the test accuracy metric
        test_acc_metric.reset_states()

        y_pred_np = np.array(y_pred).flatten()


        return test_acc.numpy(), y_pred_np
    
    

class Sequential_Models():


    def CNN(dropout_rate: float = 0.5,layer : ArrayLike = [32,64,64]):
        model = models.Sequential()

        model.add(layers.Input(shape=(28, 28, 1)))

        for current in layer:
            model.add(layers.Conv2D(current, (3, 3), 
                                    activation='relu',
                                    padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Dropout(dropout_rate))


        model.add(layers.Flatten())
        
        model.add(layers.Dense(64, 
                               activation='relu'))
        
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(1, activation='sigmoid'))

        return model



    def ResNet(dropout_rate: float = 0.5):
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
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model
    
    def MobileNet():
        pass