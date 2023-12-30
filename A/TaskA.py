"""
AMLS 1 Final Assessment - TaskA.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""



#Custom Function Imports
from Plotting import Plotting
from Models.SVM import SVM, SVM_HOG
from Models.RandomForest import RandomForest, ADABoost
from Models.NN import NN
from PreProcessing import PreProcessing

#Python Package Imports
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt


class Task_A:
    
    def __init__ (self):
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

        self.y_train = Pneumonia_train_labels
        self.y_val = Pneumonia_val_labels
        self.y_test = Pneumonia_test_labels

        print("Shape of Raw Data | Train: {} Val: {} Test: {}".format(
            self.X_train.shape,self.X_val.shape,self.X_test.shape))
        
        print("\nDATA Processing\n")

        fig, axs = Plotting().Data_Represenation(self.X_train, self.y_train, "Examples of Classes In PnumoniaMNIST Dataset")
        axs.figure.savefig('./A/Graphics/Unprocessed_A_Data.pdf')
        
        self.X_train, self.y_train = PreProcessing(self.X_train,self.y_train).New_Data(Loops=1)

        self.X_val, self.y_val = PreProcessing(self.X_val,self.y_val).New_Data(Loops=1)
        
        #self.X_test, Labels = PreProcessing(self.X_test,self.y_test).New_Data()


        self.X_train = PreProcessing.Normalisation(self.X_train)
        self.X_val = PreProcessing.Normalisation(self.X_val)
        self.X_test = PreProcessing.Normalisation(self.X_test)
        
        print("Shape of Processed Data | Train: {} Val: {} Test: {}".format(
            self.X_train.shape,self.X_val.shape,self.X_test.shape))
        
        fig, axs = Plotting().Data_Represenation(self.X_train, self.y_train,"Examples of PreProcessed Classes in PnumoniaMNIST")
        axs.figure.savefig('./A/Graphics/PreProcessed_A_Data.pdf')



    def RunSVM(self):
        image_directory = "./A/Graphics/SVM/"

        print("\nModel Selected - SVM\n")

        Model = SVM(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()

        Model.Fit()

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVM for Task A")
        axs.figure.savefig(image_directory+"SVM_Confusion_Matrix.pdf")
        Plotting().Metrics(true_labels, result)

    def RunSVM_HOG(self):
        image_directory = "./A/Graphics/SVM/"
        print("\nModel Selected - SVM w/HOG\n")

        Model = SVM_HOG(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        
        Model.SetModel()
        Model.Hog_Convert()
        Model.Fit()

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVMHOG for Task A")
        axs.figure.savefig(image_directory+"SVM_HOG_Confusion_Matrix.pdf")
        Plotting().Metrics(true_labels, result)

    def RunRF(self):
        image_directory = "./A/Graphics/RF/"
        print("\nModel Selected - RF\n")

        Model = RandomForest(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Fit()

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVM for Task A")
        axs.figure.savefig(image_directory+"RF_Confusion_Matrix.pdf")
        Plotting().Metrics(true_labels, result)
        

    def RunAdaboost(self):
        image_directory = "./A/Graphics/RF/"
        print("\nModel Selected - ADABOOST \n")

        Model = ADABoost(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Fit()
        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of SVM of test set")
        axs.figure.savefig(image_directory+"ADABoost_Confusion_Matrix.pdf")
        Plotting().Metrics(true_labels, result)



    def RunCNN(self):
        image_directory = "./A/Graphics/CNN/"

        print("\nModel Selected - CNN\n")


        learning_rate = 0.0001
        batch_size = 32
        epochs = 50
        dropout_rate = 0.25

        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

        model = ""


        Model.SetModel(model="Final",dropout_rate=dropout_rate,)
        Model.SetHyperPerameters(epochs=epochs, batchsize = batch_size, 
                                learningrate = learning_rate)
        history = Model.Train()

        if history:
            x_scale =  np.linspace(1, len(history["train_acc"]),num=len(history["train_acc"]), endpoint=True)
            fig, axs = Plotting().Line_Plot(x = x_scale, y = [history["train_acc"],history["val_acc"]],
                                            title ="Epochs Vs Accuracy for task A Final Model",
                                            x_label ="No. of Epochs", 
                                            y_label ="Accuracy", 
                                            legend = ["Train Accuracy","Validation Accuracy"])
            axs.figure.savefig(image_directory+"Final_Accuracy_plot.pdf")
        
        print("\nRESULTS\n")

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of test set for Final Model")
        axs.figure.savefig(image_directory+"Final_Confusion_Matrix.pdf")

        Plotting().Metrics(true_labels, result)
    
    def final_cross_validation(self):
        image_directory = "./A/Graphics/CNN/"

        results = {"train_acc":[],"val_acc":[]}
        learning_rate = 0.005
        batch_size = 32
        epochs = [1,2,3,4,5,6,7,8,9,10]

        dropout_rate = 0.25
        for value in epochs:
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

            t_acc_av, t_loss_av, v_acc_av, v_loss_av = Model.CrossValidation( name = "Final", loops = 5,
                                                epochs = value, batchsize = batch_size,
                                                learning_rate = learning_rate, dropout_rate = dropout_rate)
            results["train_acc"].append(t_acc_av)
            results["val_acc"].append(v_acc_av)

        fig, axs = Plotting().Line_Plot(x = epochs, y = [results["train_acc"],results["val_acc"]],
                                            legend = ["Train Accuracy","Validation Accuracy"])
        
        axs.figure.savefig(image_directory+"Final_Cross_Validation.pdf")


    def final_hyperperameter_tuning(self):  
        image_directory = "./A/Graphics/CNN/"
 
        print("Hyper Perameter Tuning Graphs")
        results = [[],[],[],[],[]]
        
        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

        base = [0.001, 32, 15, 0.25]
        learning_rate = np.logspace(-6, -1, num=10)#learning_rate = [0.000001,0.000005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
        batch_size = [2**i for i in range(0, 11)]#batch_size = [4,8,16,32,64,128,256,512,1024]
        epochs = 100
        dropout_rate = np.linspace(0, 0.9, num=10)#dropout_rate = [0,0.1,0.2,0.4,0.5,0.7,0.9]

        layers = [[64],[32,64],[32,64,128],[32,64,128,64],]
        
        for lr in learning_rate:
            print(lr)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0, dropout_rate = base[3])
            Model.SetHyperPerameters(epochs=base[2],batchsize = base[1],learningrate = lr)
            train_acc, train_loss, val_acc, val_loss = Model.train_one_off()
            
            results
            print(train_acc,train_loss,val_acc,val_loss)
            accuracy, result = Model.Test()#TEMP


        for bs in batch_size:
            print(bs)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0, dropout_rate = base[3])
            Model.SetHyperPerameters(epochs=base[2],batchsize = bs,learningrate = base[0])

            train_acc, train_loss, val_acc, val_loss = Model.train_one_off()
            print(train_acc,train_loss,val_acc,val_loss)
            accuracy, result = Model.Test()#TEMP


        for dr in dropout_rate:
            print(dr)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0,dropout_rate=dr)
            Model.SetHyperPerameters(epochs=base[2],batchsize = base[1],learningrate = base[0])

            train_acc, train_loss, val_acc, val_loss = Model.train_one_off()
            print(train_acc,train_loss,val_acc,val_loss)
            accuracy, result = Model.Test()#TEMP

        for current in layers:
            print(current)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0,dropout_rate=base[3],layer = current)
            Model.SetHyperPerameters(epochs=base[2],batchsize = base[1],learningrate = base[0])

            train_acc, train_loss, val_acc, val_loss = Model.train_one_off()
            print(train_acc,train_loss,val_acc,val_loss)
            accuracy, result = Model.Test()#TEMP

        print(epochs)
        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel(verbose=0, dropout_rate = base[3])
        Model.SetHyperPerameters(epochs=epochs,batchsize = base[1],learningrate = base[0])
        history = Model.Train()
        print(history)



Class = Task_A()


#Class.RunCNN()
#Class.final_cross_validation()

Class.final_hyperperameter_tuning()









