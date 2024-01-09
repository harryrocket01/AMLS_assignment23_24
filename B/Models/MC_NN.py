"""
AMLS 1 Final Assessment - MV_NN.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

#Custom Function Imports
from B.Models.Template import ML_Template

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

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

class MC_NN(ML_Template):
    """
    class: MC_NN

    Class containing all the functions to build, test, and train a CNN for
    MLS1 Multi class classification task.

    Attributes:
        name (str): name of model
        history (array): array containing training history
        epochs (int): number of epochs to train
        batchsize (int): size of batchs when training
        learningrate (float): learning rate hyper perameter

    Methods:
        __init__(): 
        set_model(): sets model
        set_hyper_perameters(): sets model hyper perameters
        train(): trains new model, or load existing model
        train_loop(): Tensor flow prebuilt train loop
        custom_train_loop(): Custom Built train loop
        cross_validation(): cross validates final model
        test(): tests model and calculates performance metrics
    Example:
        task_instance = MC_NN()
    """
    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)
        self.name = ""
        self.model = None

        self.history = None

        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.001


    def set_model(self,Model = "resnet", verbose: int = 1):
        """
        function: set_model
        
        Sets the model atrabute based off of string input.

        args:
            model (str): name of selected model
            verbose (int): variable if outputs should be printed
        Example:
            MC_NN().set_model()
        """
        Model = Model.lower()
        if Model == "resnet":
            self.model = CNN_Models.ResNet()
        elif Model == "resnet2":
            self.model = CNN_Models.ResNet_2()
        elif Model == "resnet50":
            self.model = CNN_Models.ResNet50()
        elif Model == "taska":
            self.model = CNN_Models.CNN_A()
        elif Model == "deep":
            self.model = CNN_Models.Deep()
        else:
            self.model = CNN_Models.Default()
        
        self.name = Model

        if verbose:
            print(self.model.summary())

    def train(self,verbose:int = 1):
        """
        function: train
        
        Trains a model if none have been saved, once trained saves the model.
        If a model has been saved, it is loaded instead.

        args:
            verbose (float): variable to indicate if a inline print should occur

        Example:
            NN().train()
        """
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
            history = self.train_loop()
            self.model.save(filename)
            
            return history

    def set_hyper_perameters(self, epochs:int =None,batch_size: int = None,learning_rate: float = None):
        """
        function: set_hyper_perameters
        
        Used to set model hyperpermaters before training

        args:
            epochs (int): input for number of desired inputs
            batchsize (int): input for desired batch size
            learningrate (float): input for desired learning rate

        Example:
            NN().set_hyper_perameters(epoch,bs,lr)
        """
        self.epochs = epochs if epochs else self.epochs
        self.batch_size = batch_size if batch_size else self.batch_size
        self.learning_rate = learning_rate if learning_rate else self.learning_rate

    def train_loop(self, verbose: str = 1):
        """
        function: train_loop
        
        Trains CNN using tensorflows pre built train_loop function. This was
        used due to the increased training speed. It operates identically to
        custom_train_loop.

        args:
            verbose (float): variable to indicate if a inline print should occur

        returns:
            history (dict): dictionary of training history, including train and validation loss and accuracy

        Example:
            NN().train_loop()
        """
        history = {}

        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss  = tf.keras.losses.CategoricalCrossentropy()

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

    def custom_trainLoop(self,verbose: int = 1):
        """
        function: custom_train_loop
        
        Custom built train loop that trains and validates cnn model. Built
        based off of the keras documentation.

        args:
            verbose (float): variable to indicate if a inline print should occur

        returns:
            history (dict): dictionary of training history, including train and validation loss and accuracy
            
        Example:
            NN().custom_train_loop()
        """
        history = {"train_acc":[],"train_loss":[],"val_loss":[],"val_acc":[],"train_time":[]}

        #Defining optomisor and loss function
        optimizer = keras.optimizers.Adam(self.learning_rate)
        loss = keras.losses.CategoricalCrossentropy()

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
    
    def test(self, verbose: int = 0) -> (int, ArrayLike):
        """
        function: test
        
        Tests model using provided training set

        args:
            verbose (float):  Indicator for if inline prints should occur

        returns:
            test_acc (float): Binary Test Accuracy value
            y_pred_np (Array): Array of predicted values
        """
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
        #convert back from one hot encoded
        y_pred_np = np.argmax(y_pred_np,  axis=1)

        return float(test_acc), y_pred_np

    def cross_validation(self, name: str = "Final", loops: int = 3, verbos: int = 2, epochs: int = 10, batchsize : int = 32, learning_rate: float = 0.001, dropout_rate: float = 0.2):
        """
        function: cross_validation
        
        Cross validates model, based of the number of passed values

        args:
            name (str):  Name of model to cross validate
            loops (int):  Number of folds to cross validate
            verbos (int):  indicator if there should be inline prints
            epochs (int):  Number of epochs for model
            batchsize (int):  Batch size of model
            learning_rate (float):  Learning rate of model
            dropout_rate (float):  Dropout rate of model

        returns:
            t_acc_av (float): average test accuracy of cross validation
            t_loss_av (float): average test loss of cross validation
            v_acc_av (float): average validation accuracy of cross validation
            v_loss_av (float): average validation loss of cross validation
        """
        results = {"train_acc":[],"train_loss":[],"val_acc":[],"val_loss":[]}

        #number of training folds
        for val_loop in range(0,loops):

            #set model
            self.set_model(model = name,dropout_rate=dropout_rate,verbose=0)
            self.set_hyper_perameters(epochs=epochs,batchsize = batchsize,learningrate = learning_rate)
            history = self.custom_train_loop(verbose=0)

            t_acc, t_loss, v_acc,v_loss = history["train_acc"][-1], history["train_loss"][-1], history["val_acc"][-1], history["val_loss"][-1]

            results["train_acc"].append(t_acc)
            results["train_loss"].append(t_loss)
            results["val_acc"].append(v_acc)
            results["val_loss"].append(v_loss)

            if verbos > 1:
                current_val = "Validation {}/{} - loss: {:.4f} - accuracy: {:.2f} - val_loss: {:.4f} - val_accuracy: {:.4f}"
                current_val = current_val.format(val_loop+1,loops,
                                            t_loss,t_acc,
                                            v_loss,v_acc)
                print(current_val)

        #calculates inal results
        t_acc_av = np.mean(results["train_acc"])
        t_loss_av = np.mean(results["train_loss"])
        v_acc_av = np.mean(results["val_acc"])
        v_loss_av = np.mean(results["val_loss"])

        if verbos >= 1:        
            final_val = "Validation Final - loss: {:.4f} - accuracy: {:.2f} - val_loss: {:.4f} - val_accuracy: {:.4f}"
            final_val = final_val.format(t_loss_av,t_acc_av,v_loss_av,v_acc_av)
            print(final_val)
        return t_acc_av, t_loss_av, v_acc_av, v_loss_av


    def visualise_layers(self):
        """
        function: visualise_layers
        
        Creates plots of feature importance for each layer of the network
        """
        for x in range(0,len(self.model.layers)-1):
            try:
                activation_model = Model(inputs=self.model.inputs, outputs=self.model.layers[x].output)
                
                img_tensor = self.X_test[0]
                img_tensor = np.expand_dims(img_tensor, axis=0)
                activation = activation_model(img_tensor)

                plt.figure(figsize=(20,20))
                for i in range(16):
                    plt.subplot(4,4,i+1)
                    plt.imshow(activation[0,:,:,i])
                plt.savefig('.\B\Graphics\CNN\Layers\Layer{}.png'.format(x))
                plt.close()
            except:
                pass


class CNN_Models():
    """
    class: CNN_Models

    Class containing all the models used within the project.
    Example models and experiments are also included. These
    can be pick and placed within the class above to be trianed,
    validated and tested.

    Methods:
        Default(): Base model reccomended by tensor flow
        CNN_A(): Final CNN used for Task A
        Deep(): CNN_A with extra conv 2D layers
        ResNet(): Final custom resnet model
        ResNet2(): Resnet with extra hidden layers
        ResNet50(): Resnet using keras resnet_50
    """
    def Default():
        """
        function: Default

        Default CNN model provided by tensor flow

        return:
            Model: Complied CNN model
        """
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
        
        model.add(layers.Dense(9,activation="softmax"))

        return model
    
    def CNN_A(dropout_rate: float = 0.25):
        """
        function: CNN_A

        CNN_A model used within task 3. Modifed to work
        within task B.

        return:
            Model: Complied CNN model
        """
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
            
        model.add(layers.Dense(9,activation="softmax"))

        return model

        
    def Deep(dropout_rate: float = 0.2):
        """
        function: Deep

        CNN_A, with extra layers to capture more data. Used
        within report.

        return:
            Model: Complied CNN model
        """
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

        model.add(layers.Conv2D(64, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
        model.add(layers.Conv2D(64, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(64, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
        model.add(layers.Conv2D(64, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(32, (3, 3), 
                                        activation='relu',
                                        padding='same',
                                        strides=1))

        model.add(layers.GlobalAveragePooling2D())
            
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(64, 
                                activation='relu'))
            
        model.add(layers.Dense(9,activation="softmax"))

        return model

    def ResNet50():
        """
        function: ResNet50

        Exploration in transfer learning. Not used in final
        report

        args:
            dropout_rate (int): Dropout rate hyperperameter
        
        return:
            Model: Complied CNN model
        """
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

    def ResNet(dropout_rate: int = 0.2):
        """
        function: ResNet

        Final Resnet Model used in report.

        return:
            Model: Complied CNN model
        """
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
        output_3 = layers.add([main_path, output_2])

        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(output_3)
        main_path = layers.Conv2D(64, 3, activation="relu", 
                                  padding="same")(main_path)
        block_4_output = layers.add([main_path, output_2])
        
        main_path = layers.Conv2D(64, (3, 3), 
                                  activation="relu")(block_4_output)

        main_path = layers.MaxPooling2D((2,2))(main_path)

        main_path = layers.GlobalAveragePooling2D()(main_path)
        main_path = layers.Dense(128, 
                                 activation="relu")(main_path)
        main_path = layers.Dropout(dropout_rate)(main_path)
        outputs = layers.Dense(9,activation="softmax")(main_path)

        model = keras.Model(inputs, outputs, name="Residual_Network")

        return model

    def ResNet_2(dropout_rate: int = 0.5):
        """
        function: ResNet2

        Final Resnet Model, with extra hidden layers. 

        return:
            Model: Complied CNN model
        """
        inputs = keras.Input(shape=(28, 28, 3))
        main_path = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        main_path = layers.Conv2D(64, (3, 3), activation="relu")(main_path)

        output_1 = layers.MaxPooling2D((2, 2))(main_path)

        main_path = layers.Conv2D(64, 3, activation="relu", padding="same", strides=1)(output_1)
        main_path = layers.Conv2D(64, 3, activation="relu", padding="same", strides=1)(main_path)
        output_2 = layers.add([main_path, output_1])

        main_path = layers.Conv2D(64, 3, activation="relu", padding="same", strides=1)(output_2)
        main_path = layers.Conv2D(64, 3, activation="relu", padding="same", strides=1)(main_path)
        output_3 = layers.add([main_path, output_2])

        main_path = layers.Conv2D(64, 3, activation="relu", padding="same", strides=1)(output_3)
        main_path = layers.Conv2D(64, 3, activation="relu", padding="same")(main_path)
        output_4 = layers.add([main_path, output_3])

        main_path = layers.Conv2D(64, 3, activation="relu", padding="same", strides=1)(output_4)
        main_path = layers.Conv2D(64, 3, activation="relu", padding="same")(main_path)
        block_5_output = layers.add([main_path, output_4])

        main_path = layers.Conv2D(64, (3, 3), activation="relu")(block_5_output)

        main_path = layers.MaxPooling2D((2, 2))(main_path)

        main_path = layers.GlobalAveragePooling2D()(main_path)
        main_path = layers.Dense(128, activation="relu")(main_path)
        main_path = layers.Dropout(dropout_rate)(main_path)
        outputs = layers.Dense(9,activation="softmax")(main_path)

        model = keras.Model(inputs, outputs, name="Residual_Network")

        return model