"""
AMLS 1 Final Assessment - NN.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""
from A.Models.Template import ML_Template


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from numpy.typing import ArrayLike
import time
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras import datasets, layers, models,optimizers
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

class NN(ML_Template):
    """
    class: NN

    Class containing all the functions to build, test, and train a CNN for
    MLS1 Binary classification task.

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
        train_one_off(): trains without saving model
        train_loop(): Tensor flow prebuilt train loop
        custom_train_loop(): Custom Built train loop
        cross_validation(): cross validates final model
        test(): tests model
    Example:
        task_instance = NN()
    """

    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)
        self.name = ""
        self.model = None

        self.history = None

        self.epochs = 25
        self.batchsize = 32
        self.learningrate = 0.0001


    def set_model(self,model = "final",verbose: int = 1,dropout_rate: float = 0.5,) -> None:
        """
        function: set_model
        
        Sets the model atrabute based off of string input.

        args:
            model (str): name of selected model
            verbose (int): variable if outputs should be printed
            dropout_rate (float): drop out hyper perameter

        Example:
            NN().set_model()
        """

        model = model.lower()
        if model == "resnet":
            self.model = Sequential_Models.ResNet(dropout_rate = dropout_rate)
        elif model =="manylayers":
            self.model = Sequential_Models.CNN(dropout_rate=dropout_rate, layer=[32, 64, 128, 256,256,256, 128, 64])
        elif model =="alt":
            self.model = Sequential_Models.CNN(dropout_rate=dropout_rate, layer= [32,64,128,32,128,32,128,64])
        elif model =="final":
            self.model = Sequential_Models.CNN_final(dropout_rate=dropout_rate)
        else:
            self.model = Sequential_Models.CNN_final(dropout_rate=dropout_rate)
        
        self.name = model

        if verbose:
            print(self.model.summary())

    def set_hyper_perameters(self, epochs:int =None,batchsize: int = None,learningrate: float = None):
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
        self.batchsize = batchsize if batchsize else self.batchsize
        self.learningrate = learningrate if learningrate else self.learningrate




    def train(self,verbose: int = 1) -> dict:
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
        filename = 'A\Models\PreTrainedModels\{}Model.keras'.format(self.name)
        try:
            self.model = tf.keras.models.load_model(filename)            

            if verbose:
                print("\nPre Trained Model Loaded - {}Model.keras".format(self.name))
            return None
        except:
            #Trains if there is no file, then saves
            if verbose:
                print("No Pre Trained Model, Training Model - {}Model.keras\n".format(self.name))
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            history = self.custom_train_loop()
            self.model.save(filename)
            return history

    def train_one_off(self,verbose: int = 0):
        """
        function: train_one_off
        
        Utility function that performs a one off train loop that does not save model

        args:
            verbose (float): variable to indicate if a inline print should occur

        Example:
            NN().train_one_off()
        """
        history = self.custom_train_loop(verbose=verbose)

        return history["train_acc"][-1], history["train_loss"][-1], history["val_acc"][-1], history["val_loss"][-1]



    def train_loop(self,verbose: int = 1) -> dict:
        """
        function: train_loop
        
        Trains CNN using tensorflows pre built train_loop function.

        args:
            verbose (float): variable to indicate if a inline print should occur

        returns:
            history (dict): dictionary of training history, including train and validation loss and accuracy

        Example:
            NN().train_loop()
        """
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

    def custom_train_loop(self,verbose: int = 1) -> dict:
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


    def test(self, verbose: int = 1) -> (int, ArrayLike):
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
        test_dataset = test_dataset.batch(self.batchsize) 

        test_acc_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5) 
        y_pred = []

        #Loop over the test set
        for x_batch_test, y_batch_test in test_dataset:
            test_logits = self.model(x_batch_test, training=False)
            batch_pred = (test_logits >= 0.5).numpy().astype(int)
            test_acc_metric.update_state(y_batch_test, test_logits)
            y_pred.extend(batch_pred)

        test_acc = test_acc_metric.result()
        if verbose:
            print("Test accuracy: %.4f" % (float(test_acc)))

        #Reset the test accuracy metric
        test_acc_metric.reset_states()

        y_pred_np = np.array(y_pred).flatten()

        return test_acc.numpy(), y_pred_np
    
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
                plt.savefig('.\A.\Graphics\CNN\Layers\Layer{}.png'.format(x))
                plt.close()
            except:
                pass

    
    

class Sequential_Models():
    """
    class: Sequential_Models

    Class containing various models used within the project.
    Example models and experiments are also included

    Methods:
        CNN(): Customised and scalable CNN model
        CNN_final(): Final CNN used for Task A
        ResNet(): Resnet using keras resnet_50
    """

    def CNN(dropout_rate: float = 0.5,layer : ArrayLike = [32,64,64]):
        """
        function: CNN

        Customised CNN that can be modifed through variables. Used to explore 
        filter size and shape

        args:
            dropout_rate (int): Drop out rate hyper perameter
            layer (ArrayLike): Filter size of sequential Conv2D layers
        
        return:
            Model: Complied CNN model

        """
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

    def CNN_final(dropout_rate: float = 0.25):
        """
        function: CNN_final

        Final Model used for Binary Classifcation Task

        args:
            dropout_rate (int): Dropout rate hyperperameter
        
        return:
            Model: Complied CNN model
        """
        model = models.Sequential()

        model.add(layers.Input(shape=(28, 28, 1)))
                
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
            
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def ResNet(dropout_rate: float = 0.5):
        """
        function: ResNet

        Exploration in transfer learning. Not used in final
        report

        args:
            dropout_rate (int): Dropout rate hyperperameter
        
        return:
            Model: Complied CNN model
        """
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
    