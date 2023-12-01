
#import subprocess
#import sys
#subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from Plotting import Plotting

import numpy as np

from Models.SVM import SVM, SVM_HOG
from Models.RandomForest import RandomForest, ADABoost
from Models.NN import NN


class Task_A:
    
    def __init__ (self):

        directory = "Datasets\pneumoniamnist.npz"

        PneumoniaMNIST = np.load(directory)


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


        n_train_samples = len(Pneumonia_train_labels)
        n_val_samples = len(Pneumonia_val_labels)
        n_test_samples = len(Pneumonia_test_labels)


        #Reshape and flatten images, devide in to train val and test
        self.X_train = Pneumonia_train_images.reshape((n_train_samples, 28, 28, 1))
        self.X_val = Pneumonia_val_images.reshape((n_val_samples, 28, 28, 1))
        self.X_test = Pneumonia_test_images.reshape((n_test_samples, 28, 28, 1))

        self.y_train = Pneumonia_train_labels
        self.y_val = Pneumonia_val_labels
        self.y_test = Pneumonia_test_labels

        #Plotting.Data_Represenation(self.X_train,self.y_train)

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


    def RunNN(self):
        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)

        Model.SetModel(Model="cnn")
        Model.Train()
        Model.Test()



Class = Task_A()
#Class.RunSVM()
#Class.RunSVM_HOG()

Class.RunRF()
Class.RunAdaboost()

#Class.RunNN()









