




#import subprocess
#import sys
#subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
from PreProcessing import PreProcessing

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from Plotting import Plotting

import numpy as np

from Models.MV_NN import NN

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
        
        #print(self.y_train)
        #self.X_test = PreProcessing(self.X_test,self.y_test).data_augmentation()
        #self.X_val = PreProcessing(self.X_val,self.y_val).data_augmentation()
        #self.X_train = PreProcessing(self.X_train,self.y_train).data_augmentation()


        self.X_train = PreProcessing.normalisation(self.X_train)
        self.X_val = PreProcessing.normalisation(self.X_val)
        self.X_test = PreProcessing.normalisation(self.X_test)

        Plotting.Data_Represenation(self.X_train,Path_train_labels)

        print(self.X_train.shape,self.X_val.shape,self.X_test.shape)

    def RunSVM(self):
        pass


    def RunNN(self):
        Model = NN(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test)
        print(Model.X_test.shape)
        Model.SetModel()
        Model.Train()
        Model.Test()



Class = Task_B()
Class.RunNN()

