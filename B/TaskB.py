from Plotting import Plotting

import numpy as np

from Multivariate_Models import MV_Models

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Task_A:
    pass

class MV_Classification_Base:
    
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

        self.y_train = Path_train_labels
        self.y_val = Path_val_labels
        self.y_test = Path_test_labels

        #Plotting.Data_Represenation( self.X_train, self.y_train)

        MV_Models(self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test).CNN(epochs=5)


if __name__ == '__main__':

    MV_Classification_Base()








