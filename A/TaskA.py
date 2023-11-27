
from Plotting import Plotting

import numpy as np

class Task_A:
    pass

class Binary_Classification_Base:
    
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
        X_train = Pneumonia_train_images.reshape((n_train_samples, 28, 28, 1))
        X_val = Pneumonia_val_images.reshape((n_val_samples, 28, 28, 1))
        X_test = Pneumonia_test_images.reshape((n_test_samples, 28, 28, 1))

        y_train = Pneumonia_train_labels
        y_val = Pneumonia_val_labels
        y_test = Pneumonia_test_labels

        Plotting.Data_Represenation(X_train,y_train)



Binary_Classification_Base()








