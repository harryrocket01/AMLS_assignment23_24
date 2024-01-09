"""
AMLS 1 Final Assessment - TaskB.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""
#Custom Function Imports
from B.PreProcessing import PreProcessing
from B.Plotting import Plotting
from B.Models.MC_NN import MC_NN

#Python Package Imports
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import matplotlib.pyplot as plt

class TaskB:
    """
    class: TaskB

    This class contains the key functions used within TASK A of the MLS 2023/4 project

    Attributes:
        X_train (numpy.ndarray): Training images dataset
        X_val (numpy.ndarray): Validation images dataset
        X_test (numpy.ndarray): Test images dataset
        y_train (numpy.ndarray): Training labels
        y_val (numpy.ndarray): Validation labels
        y_test (numpy.ndarray): Test labels

    Methods:
        __init__(): Initializes the TaskB class, imports and processes the PneumoniaMNIST dataset
        run_cnn(): Creates, train and validates CNN model - Final Model
        final_cross_validation(): Cross Validates final CNN model
        learning_rate_effect(): creates plot of effect of learning rate
        dr_bs_tuning(): Creates plot to tune batch size and dropout rate
    Example:
        task_instance = TaskA()
    """
    def __init__ (self):
        """
        __init__

        The function imports the dataset, extracts training, validation, and test data,
        reshapes and flattens the images, sets labels, performs data representation plotting,
        applies data augmentation, and normalizes the data ready for use.

        Example:
            task_instance = TaskB()
        """
        directory = "Datasets\pathmnist.npz"

        PathMNIST = np.load(directory)

        print("Keys in PathMNIST: ", list(PathMNIST.keys()))

        #Extract Pathmnist

        #Train
        Path_train_images = PathMNIST['train_images']
        Path_train_labels = PathMNIST['train_labels']

        #Validation
        Path_val_images = PathMNIST['val_images']
        Path_val_labels = PathMNIST['val_labels']

        #Test
        Path_test_images = PathMNIST['test_images']
        Path_test_labels = PathMNIST['test_labels']

        n_train_samples = len(Path_train_images)
        n_val_samples = len(Path_val_images)
        n_test_samples = len(Path_test_images)

        #Reshape and flatten images, devide in to train val and test
        self.X_train = Path_train_images.reshape((n_train_samples, 28, 28, 3))
        self.X_val = Path_val_images.reshape((n_val_samples, 28, 28, 3))  
        self.X_test = Path_test_images.reshape((n_test_samples, 28, 28, 3))

        self.y_train,self.y_val,self.y_test = Path_train_labels,Path_val_labels,Path_test_labels
        self.y_train,self.y_val,self.y_test = PreProcessing.one_hot_encode(Path_train_labels,Path_val_labels,Path_test_labels)

        fig, axs = Plotting().Data_Represenation(self.X_train, Path_train_labels,"Examples of Classes in PathMNIST")
        fig.savefig('./B/Graphics/UnProcessed_B_Data.pdf')

        self.X_train = PreProcessing.normalisation(self.X_train)
        self.X_val = PreProcessing.normalisation(self.X_val)
        self.X_test = PreProcessing.normalisation(self.X_test)

        fig, axs = Plotting().Data_Represenation(self.X_train, Path_train_labels,"Examples of PreProcessed Classes in PathMNIST")
        fig.savefig('./B/Graphics/PreProcessed_B_Data.pdf')

    def run_nn(self, model_name: str = "resnet"):
        """
        function: run_nn
        
        Creates, trains and validates random fianl CNN model for Task A.
        This is the final model chosen for this task, and produces a graph for 
        epochs against accuracy & loss, along with a confusion matrix of the result.
        Example:
            task_instance = TaskA()
            task_instance.run_cnn()
        """
        image_directory = "./B/Graphics/CNN/"
        print("\n____________ TASK B ____________\n")

        epochs = 20
        learning_rate = 0.001
        batch_size = 32

        model = MC_NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        model.set_model(Model = model_name)
        model.set_hyper_perameters(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

        history = model.train()

        if history:
            x_scale =  np.linspace(1, len(history["train_acc"]),num=len(history["train_acc"]), endpoint=True)
            fig, axs = Plotting().acc_loss_plot(history["train_acc"],history["train_loss"],
                                                history["val_acc"],history["val_loss"],
                                                x_scale,"Accuracy-Loss Plot for Final Model")

            axs.figure.savefig(image_directory+model_name+"_Accuracy_plot.pdf")
        
        print("\nRESULTS\n")
        
        accuracy, result = model.test()

        true_labels = np.argmax(self.y_test,  axis=1)

        Plotting().metrics(true_labels,result)

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of "+model_name+" on the test set")
        axs.figure.savefig(image_directory+model_name+"_Confusion_Matrix.pdf")


    def final_cross_validation(self):
        """
        function: final_cross_validation
        
        Creates, trains and validates random fianl CNN model for Task B.
        This is the final model chosen for this task, and produces a graph for 
        epochs against accuracy & loss, along with a confusion matrix of the result.
        Example:
            task_instance = TaskB()
            task_instance.final_cross_validation()
        """
        image_directory = "./B/Graphics/CNN/"

        results = {"train_acc":[],"val_acc":[]}
        epochs = 20
        learning_rate = 0.001
        batch_size = 32
        dropout_rate = 0.25

        for value in epochs:
            Model = MC_NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.set_hyper_perameters(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

            t_acc_av, t_loss_av, v_acc_av, v_loss_av = Model.cross_validation( name = "Final", loops = 5,
                                                epochs = value, batchsize = batch_size,
                                                learning_rate = learning_rate, dropout_rate = dropout_rate)
            results["train_acc"].append(t_acc_av)
            results["val_acc"].append(v_acc_av)

        fig, axs = Plotting().line_plot(x = epochs, y = [results["train_acc"],results["val_acc"]],
                                            legend = ["Train Accuracy","Validation Accuracy"])
        axs.figure.savefig(image_directory+"Final_Cross_Validation.pdf")

