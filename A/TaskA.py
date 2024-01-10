"""
AMLS 1 Final Assessment - TaskA.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

#Custom Function Imports
from A.Plotting import Plotting
from A.Models.SVM import SVM, SVM_HOG
from A.Models.RandomForest import RandomForest, ADABoost
from A.Models.NN import NN
from A.PreProcessing import PreProcessing

#Python Package Imports
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt

class TaskA:
    """
    class: TaskA

    This class contains the key functions used within TASK A of the MLS 2023/4 project

    Attributes:
        X_train (numpy.ndarray): Training images dataset
        X_val (numpy.ndarray): Validation images dataset
        X_test (numpy.ndarray): Test images dataset
        y_train (numpy.ndarray): Training labels
        y_val (numpy.ndarray): Validation labels
        y_test (numpy.ndarray): Test labels

    Methods:
        __init__(): Initializes the TaskA class, imports and processes the PneumoniaMNIST dataset
        run_svm(): Creates, train and validates SVM model
        run_svm_hog(): Creates, train and validates SVM model with HOG feature extraction
        run_rf(): Creates, train and validates Random Forrest Model
        run_adaboost(): Creates, train and validates Adaboost model
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
            task_instance = TaskA()
        """
        print("____________ TASK A ____________\n")
        print("\nDATA IMPORTS\n")

        directory = "Datasets\pneumoniamnist.npz"

        PneumoniaMNIST = np.load(directory)

        print("Imported Data from {}".format(directory))
        print("Keys in PneumoniaMNIST: ", list(PneumoniaMNIST.keys()))

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

        #Reshape and flatten images, devide in to train val and test
        self.X_train = Pneumonia_train_images.reshape((len(Pneumonia_train_labels), 28, 28, 1))
        self.X_val = Pneumonia_val_images.reshape((len(Pneumonia_val_labels), 28, 28, 1))
        self.X_test = Pneumonia_test_images.reshape((len(Pneumonia_test_labels), 28, 28, 1))

        #set labels
        self.y_train = Pneumonia_train_labels
        self.y_val = Pneumonia_val_labels
        self.y_test = Pneumonia_test_labels

        print("Shape of Raw Data | Train: {} Val: {} Test: {}".format(
            self.X_train.shape,self.X_val.shape,self.X_test.shape))
        print("\nDATA Processing\n")

        #plot Data_Represenation
        fig, axs = Plotting().data_represenation(self.X_train, self.y_train, "Examples of Classes In PnumoniaMNIST Dataset")
        axs.figure.savefig('./A/Graphics/Unprocessed_A_Data.pdf')
        
        #Data augmentation
        self.X_train, self.y_train = PreProcessing(self.X_train,self.y_train).new_Data(loops=1)
        self.X_val, self.y_val = PreProcessing(self.X_val,self.y_val).new_Data(loops=1)
        #self.X_test, self.y_test = PreProcessing(self.X_test,self.y_test).new_Data(loops=1)

        #data normalisation
        self.X_train = PreProcessing.normalisation(self.X_train)
        self.X_val = PreProcessing.normalisation(self.X_val)
        self.X_test = PreProcessing.normalisation(self.X_test)
        
        print("Shape of Processed Data | Train: {} Val: {} Test: {}".format(
            self.X_train.shape,self.X_val.shape,self.X_test.shape))
        
        fig, axs = Plotting().data_represenation(self.X_train, self.y_train,"Examples of PreProcessed Classes in PnumoniaMNIST")
        axs.figure.savefig('./A/Graphics/PreProcessed_A_Data.pdf')


    def run_svm(self):
        """
        function: run_svm
        Creates, trains and validates Support Vector Machine (SVM) model on for Task A.

        Example:
            task_instance = TaskA()
            task_instance.run_svm()
        """
        image_directory = "./A/Graphics/SVM/"
        print("\nModel Selected - SVM\n")

        Model = SVM(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Fit()

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().confusion_matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVM for Task A")
        axs.figure.savefig(image_directory+"SVM_Confusion_Matrix.pdf")

        Plotting().metrics(true_labels, result)


    def run_svm_hog(self):
        """
        function: run_svm_hog
        
        Creates, trains and validates Support Vector Machine (SVM) model 
        with hog feature extraction on for Task A.

        Example:
            task_instance = TaskA()
            task_instance.run_svm_hog()
        """

        image_directory = "./A/Graphics/SVM/"
        print("\nModel Selected - SVM w/HOG\n")

        Model = SVM_HOG(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Hog_Convert()
        Model.Fit()

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().confusion_matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVMHOG for Task A")
        axs.figure.savefig(image_directory+"SVM_HOG_Confusion_Matrix.pdf")

        Plotting().metrics(true_labels, result)


    def run_rf(self):
        """
        function: run_rf
        
        Creates, trains and validates random forrest model for Task A.

        Example:
            task_instance = TaskA()
            task_instance.run_rf()
        """
        image_directory = "./A/Graphics/RF/"
        print("\nModel Selected - RF\n")

        Model = RandomForest(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Fit()

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()
        fig, axs = Plotting().confusion_matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVM for Task A")
        axs.figure.savefig(image_directory+"RF_Confusion_Matrix.pdf")

        Plotting().metrics(true_labels, result)
        

    def run_adaboost(self):
        """
        function: run_adaboost
        
        Creates, trains and validates Adaboost model for Task A.

        Example:
            task_instance = TaskA()
            task_instance.run_adaboost()
        """
        image_directory = "./A/Graphics/RF/"
        print("\nModel Selected - ADABOOST \n")

        Model = ADABoost(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Fit()
        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().confusion_matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVM of test set")
        axs.figure.savefig(image_directory+"ADABoost_Confusion_Matrix.pdf")

        Plotting().metrics(true_labels, result)


    def run_cnn(self):
        """
        function: run_cnn
        
        Creates, trains and validates random fianl CNN model for Task A.
        This is the final model chosen for this task, and produces a graph for 
        epochs against accuracy & loss, along with a confusion matrix of the result.
        Example:
            task_instance = TaskA()
            task_instance.run_cnn()
        """
        image_directory = "./A/Graphics/CNN/"
        print("\nModel Selected - CNN\n")

        #Hyper-perameters
        learning_rate = 0.0001
        batch_size = 40
        epochs = 100
        dropout_rate = 0.25
        model_name = "final"

        model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        model.set_model(model=model_name,dropout_rate=dropout_rate,)
        model.set_hyper_perameters(epochs=epochs, batchsize = batch_size, 
                                learningrate = learning_rate)
        history = model.train()

        if history:
            x_scale =  np.linspace(1, len(history["train_acc"]),num=len(history["train_acc"]), endpoint=True)
            fig, axs = Plotting().acc_loss_plot(history["train_acc"],history["train_loss"],
                                                history["val_acc"],history["val_loss"],
                                                x_scale,"Accuracy-Loss Plot for Final Model")
            axs.figure.savefig(image_directory+"Final_Accuracy_plot.pdf")
        
        print("\nRESULTS\n")
        accuracy, result = model.test()
        true_labels = np.array(self.y_test).flatten()
        fig, axs = Plotting().confusion_matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of test set for Final Model")
        axs.figure.savefig(image_directory+"Final_Confusion_Matrix.pdf")
        Plotting().metrics(true_labels, result)
    

    def final_cross_validation(self):
        """
        function: final_cross_validation
        
        Creates, trains and validates random fianl CNN model for Task A.
        This is the final model chosen for this task, and produces a graph for 
        epochs against accuracy & loss, along with a confusion matrix of the result.
        Example:
            task_instance = TaskA()
            task_instance.final_cross_validation()
        """
        image_directory = "./A/Graphics/CNN/"

        results = {"train_acc":[],"val_acc":[]}
        learning_rate = 0.0001
        batch_size = 32
        epochs = [1,2,3,4,5,6,7,8,9,10]

        dropout_rate = 0.25
        for value in epochs:
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

            t_acc_av, t_loss_av, v_acc_av, v_loss_av = Model.cross_validation( name = "Final", loops = 5,
                                                epochs = value, batchsize = batch_size,
                                                learning_rate = learning_rate, dropout_rate = dropout_rate)
            results["train_acc"].append(t_acc_av)
            results["val_acc"].append(v_acc_av)

        fig, axs = Plotting().line_plot(x = epochs, y = [results["train_acc"],results["val_acc"]],
                                            legend = ["Train Accuracy","Validation Accuracy"])
        axs.figure.savefig(image_directory+"Final_Cross_Validation.pdf")


    def learning_rate_effect(self):  
        """
        function: learning_rate_effect
        
        Creates plot of effect of various learning rates within the
        final CNN model
        Example:
            task_instance = TaskA()
            task_instance.learning_rate_effect()
        """
        image_directory = "./A/Graphics/CNN/"

        lr = [0.01, 0.0001,0.0000001]
        epochs = 25
        x_scale =  np.linspace(1, epochs,num=epochs, endpoint=True)

        #LR = 0.01
        Model1 = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model1.set_model(verbose=0,dropout_rate=0.25)
        Model1.set_hyper_perameters(learningrate = lr[0], epochs=epochs)
        history1 = Model1.TrainLoop()

        #LR = 0.0001
        Model2 = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model2.set_model(verbose=0,dropout_rate=0.25)
        Model2.set_hyper_perameters(learningrate = lr[1], epochs=epochs)
        history2 = Model2.TrainLoop()

        #LR = 0.0000001
        Model3 = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model3.set_model(verbose=0,dropout_rate=0.25)
        Model3.set_hyper_perameters(learningrate = lr[2], epochs=epochs)
        history3 = Model3.TrainLoop()

        fig, axs = Plotting().learningrate_plot(history1["train_acc"], history1["val_acc"], 
                          history2["train_acc"], history2["val_acc"],
                          history3["train_acc"], history3["val_acc"],
                          x_scale,lr)
        fig.savefig(image_directory+"learning_rate.pdf")


    def dr_bs_tuning(self):  
        """
        function: dr_bs_tuning
        
        Creates plot of effect of various learning rates within the
        final CNN model.
        Example:
            task_instance = TaskA()
            task_instance.dr_bs_tuning()
        """
        image_directory = "./A/Graphics/CNN/"
 
        print("Hyper Perameter Tuning Graphs")
        results = [[],[],[],[]]
        
        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

        base = [0.001, 32, 15, 0.25]
        batch_size = [2**i for i in range(0, 10)]#batch_size = [4,8,16,32,64,128,256,512,1024]
        dropout_rate = np.linspace(0, 0.9, num=10)#dropout_rate = [0,0.1,0.2,0.4,0.5,0.7,0.9]
        
        #Loop for Batch size changes
        for bs in batch_size:
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.set_model(verbose=0, dropout_rate = base[3])
            Model.set_hyper_perameters(epochs=base[2],batchsize = bs,learningrate = base[0])
            train_acc, train_loss, val_acc, val_loss = Model.train_one_off()
            results[0].append(train_acc)
            results[1].append(val_acc)

        #Loop for dropout rate changes
        for dr in dropout_rate:
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.set_model(verbose=0,dropout_rate=dr)
            Model.set_hyper_perameters(epochs=base[2],batchsize = base[1],learningrate = base[0])

            train_acc, train_loss, val_acc, val_loss = Model.train_one_off()
            results[2].append(train_acc)
            results[3].append(val_acc)

        fig, axs = Plotting().dr_bs_plot(results[0],results[1],
                          results[2],results[3],
                          batch_size,dropout_rate)
        fig.savefig(image_directory+"dr_bs_tuning.pdf")



