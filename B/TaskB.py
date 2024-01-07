"""
AMLS 1 Final Assessment - TaskA.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""
#Custom Function Imports
from PreProcessing import PreProcessing
from Plotting import Plotting
from Models.MV_NN import MV_CNN

#Python Package Imports
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import matplotlib.pyplot as plt

class Task_B:
    
    def __init__ (self):
        directory = "Datasets\pathmnist.npz"

        PathMNIST = np.load(directory)

        print("Keys in PathMNIST: ", list(PathMNIST.keys()))




        #Extract Pneumonia

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




    def RunSVM(self):
        pass


    def RunNN(self, model_name: str = "resnet"):
        image_directory = "./B/Graphics/CNN/"

        epochs = 20
        learning_rate = 0.001
        batch_size = 32

        Model = MV_CNN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        print(Model.X_test.shape)
        Model.SetModel(Model = model_name)
        Model.SetHyperPerameters(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

        history = Model.Train()
        print(history)
        if history:
            x_scale =  np.linspace(1, len(history["train_acc"]),num=len(history["train_acc"]), endpoint=True)
            fig, axs = Plotting().acc_loss_plot(history["train_acc"],history["train_loss"],
                                                history["val_acc"],history["val_loss"],
                                                x_scale,"Accuracy-Loss Plot for Final Model")

            axs.figure.savefig(image_directory+model_name+"_Accuracy_plot.pdf")
        
        print("\nRESULTS\n")
        
        accuracy, result = Model.Test()
        print(self.y_test)

        true_labels = np.argmax(self.y_test,  axis=1)

        Plotting().metrics(true_labels,result)

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of "+model_name+" on the test set")
        axs.figure.savefig(image_directory+model_name+"_Confusion_Matrix.pdf")



Class = Task_B()
Class.RunNN()

