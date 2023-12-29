import subprocess
import sys
#subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn "])

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from Plotting import Plotting

import numpy as np

from Models.SVM import SVM, SVM_HOG
from Models.RandomForest import RandomForest, ADABoost
from Models.NN import NN
from PreProcessing import PreProcessing
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

        #self.X_val, Labels = PreProcessing(self.X_val,self.y_val).New_Data()
        #self.X_test, Labels = PreProcessing(self.X_test,self.y_test).New_Data()


        self.X_train = PreProcessing.Normalisation(self.X_train)
        self.X_val = PreProcessing.Normalisation(self.X_val)
        self.X_test = PreProcessing.Normalisation(self.X_test)
        
        print("Shape of Processed Data | Train: {} Val: {} Test: {}".format(
            self.X_train.shape,self.X_val.shape,self.X_test.shape))
        
        fig, axs = Plotting().Data_Represenation(self.X_train, self.y_train,"Examples of PreProcessed Classes in PnumoniaMNIST")
        axs.figure.savefig('./A/Graphics/PreProcessed_A_Data.pdf')



    def RunSVM(self):
        Model = SVM(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        
        #Model.BayOpt()

        Model.SetModel()
        Model.Fit()
        Model.Test()

    def RunSVM_HOG(self):
        Model = SVM_HOG(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        
        Model.SetModel()
        Model.Hog_Convert()
        Model.Fit()
        Model.Test()

    def RunRF(self):
        Model = RandomForest(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Fit()
        Model.Test()

    def RunAdaboost(self):
        Model = ADABoost(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel()
        Model.Fit()
        Model.Test()


    def RunFinalNN(self):
        print("\nModel Selected - CNN\n")


        learning_rate = 0.001
        batch_stize = 16
        epochs = 5
        dropout_rate = 0.2

        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

        model = ""


        Model.SetModel(dropout_rate=dropout_rate)

        history = Model.Train(epochs=epochs,batchsize = batch_stize,learningrate = learning_rate)

        if history:
            x_scale =  np.linspace(1, len(history["train_acc"]),num=len(history["train_acc"]), endpoint=True)
            fig, axs = Plotting().Line_Plot(x = x_scale, y = [history["train_acc"],history["val_acc"]],
                                            title ="", 
                                            x_label ="", 
                                            y_label ="", 
                                            legend = ["Train Accuracy","Validation Accuracy"])
            axs.figure.savefig('./A/Graphics/Final_Accuracy_plot.pdf')
        
        print("\nRESULTS\n")

        accuracy, result = Model.Test()
        true_labels = np.array(self.y_test).flatten()

        fig, axs = Plotting().Confusion_Matrix(true_labels = true_labels, pred_labels=result,
                                                title= "Confusion Matrix of Final Model for Task A")
        axs.figure.savefig('./A/Graphics/Final_Confusion_Matrix.pdf')

        Plotting().Metrics(true_labels, result)
    


    def hyperperameter_tuning(self):   
        print("Hyper Perameter Tuning Graphs")
        results = [[],[],[],[],[]]
        
        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

        base = [0.005, 32, 25, 0.25]
        learning_rate = np.logspace(-6, -1, num=40)#learning_rate = [0.000001,0.000005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
        batch_size = [2**i for i in range(0, 11)]#batch_size = [4,8,16,32,64,128,256,512,1024]
        epochs = 100
        dropout_rate = np.linspace(0, 0.9, num=40)#dropout_rate = [0,0.1,0.2,0.4,0.5,0.7,0.9]

        layers = [[64],[32,64],[32,64,128],[32,64,128,64],]
        
        for lr in learning_rate:
            print(lr)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0, dropout_rate = base[3])
            train_acc, train_loss, val_acc, val_loss = Model.train_one_off(epochs=base[2],
                                                                           batchsize = base[1],
                                                                           learningrate = lr)
            
            results
            print(train_acc,train_loss,val_acc,val_loss)


        for bs in batch_size:
            print(bs)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0, dropout_rate = base[3])
            train_acc, train_loss, val_acc, val_loss = Model.train_one_off(epochs=base[2],
                                                                           batchsize = bs,
                                                                           learningrate = base[0])
            print(train_acc,train_loss,val_acc,val_loss)
        

        for dr in dropout_rate:
            print(dr)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0,dropout_rate=dr)
            train_acc, train_loss, val_acc, val_loss = Model.train_one_off(epochs=base[2],
                                                                           batchsize = base[1],
                                                                           learningrate = base[0])
            print(train_acc,train_loss,val_acc,val_loss)

        for current in layers:
            print(current)
            Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
            Model.SetModel(verbose=0,dropout_rate=base[3],layer = current)
            train_acc, train_loss, val_acc, val_loss = Model.train_one_off(epochs=base[2],
                                                                           batchsize = base[1],
                                                                           learningrate = base[0])
            print(train_acc,train_loss,val_acc,val_loss)
        
        print(epochs)
        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        Model.SetModel(verbose=0, dropout_rate = base[3])
        history = Model.Train(epochs=epochs,batchsize = base[1],learningrate = base[0])
        print(history)



Class = Task_A()
#Class.RunSVM()
#Class.RunSVM_HOG()

#Class.RunRF()
#Class.RunAdaboost()

#Class.RunFinalNN()

Class.hyperperameter_tuning()









