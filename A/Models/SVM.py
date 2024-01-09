from A.Models.Template import ML_Template

import numpy as np
from numpy.typing import ArrayLike

import sklearn
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from scipy import ndimage
import matplotlib.pyplot as plt

class SVM(ML_Template):
    """
    class: SVM

    Class that contains all the functions to complie, fit and 
    evaluate a SVM model.

    Attributes:
        c (float): Regularisation hyper perameter
        gamma (float): Hyper perameter that defines increment influence
        kernel (str): type of kernel used within SVM

    Methods:
        __init__(): 
        SetModel(): sets model
        Fit(): fits training set to model
        Test(): tests model againt validation and test set
        gs_cv(): uses gridsearch cv to optomise hyper perameters

    Example:
        SVM()
    """

    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)

        self.X_train=self.X_train.reshape(len(self.y_train), 28*28)
        self.X_val=self.X_val.reshape(len(self.y_val), 28*28)
        self.X_test=self.X_test.reshape(len(self.y_test), 28*28)

        self.y_train = self.y_train.ravel()
        self.y_val = self.y_val.ravel()
        self.y_test = self.y_test.ravel()

        self.C = 1
        self.gamma = 0.01
        self.kernel = 'rbf'

    def SetModel(self,kernel = None,C=None,gamma = None):
        """
        function: SetModel
        
        Sets the model atrabute based off of hyper perameter input.

        args:
            kernel (str): name of selected kernal
            C (float): value for C hyper perameter
            gamma (float): value for gamma hyper perameter

        Example:
            SVM().SetModel()
        """
        if kernel:
            self.kernel = kernel
        if C:
            self.C = C
        if gamma:
            self.gamma
        #sets given model
        SVM_Model = svm.SVC(kernel=self.kernel, C=self.C, gamma = self.gamma,verbose=1)
        self.Model = SVM_Model

    def Fit(self):
        """
        function: Fit
        
        Fits set model to the training set, provides validation
        accuracy score
        
        return:
            test_acc (float) : test accuracy score
            y_pred (ArrayLike): Array of predicted labels

        Example:
            SVM().Fit()
        """
        self.Model.fit(self.X_train,self.y_train)
        y_pred = self.Model.predict(self.X_val)
        val_acc = metrics.accuracy_score(self.y_val, y_pred)

        print("Validation Accuracy:",val_acc)

        return val_acc, y_pred

    def Test(self):
        """
        function: Test
        
        Tests a prefit model against the test set

        return:
            test_acc (float) : test accuracy score
            y_pred (ArrayLike): Array of predicted labels

        Example:
            SVM().Test()
        """
        y_pred = self.Model.predict(self.X_test)
        test_acc = metrics.accuracy_score(self.y_test, y_pred)

        print("Accuracy:", test_acc)

        return test_acc, y_pred


        

    def gs_cv(self,Crange = [0.1, 1, 10, 100],gammarange = [1, 0.1, 0.01, 0.001],kernalrange =['linear', 'poly', 'rbf', 'sigmoid'] ):
        
        params  = {
            'C': Crange,
            'gamma': gammarange,
            'kernel': kernalrange,
        }

        grid_search = GridSearchCV(SVC(), params, cv=5,verbose=10)

        # Fit the model
        grid_search.fit(self.X_train, self.y_train)

        # Best parameters and model
        print("Best Parameters:", grid_search.best_params_)




class SVM_HOG(ML_Template):
    """
    class: SVM_HOG

    Class that contains all the functions to complie, fit and 
    evaluate a SVM model.

    Attributes:
        c (float): Regularisation hyper perameter
        gamma (float): Hyper perameter that defines increment influence
        kernel (str): type of kernel used within SVM
        Hog_train (ArrayLike): HOG training set
        Hog_val (ArrayLike): Hog Validation set
        Hog_test (ArrayLike): Hog test set

    Methods:
        __init__(): 
        SetModel(): sets model
        Fit(): fits training set to model
        Test(): tests model againt validation and test set
        HOG(): computes single images features through HOG
        Hog_Batch(): computes set of images features through HOG
        Hog_Convert(): converts all sets to HOG

    Example:
        SVM_HOG()
    """


    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)

        self.X_train=self.X_train.squeeze(axis=-1)
        self.X_val=self.X_val.squeeze(axis=-1)
        self.X_test=self.X_test.squeeze(axis=-1)


        self.y_train = self.y_train.ravel()
        self.y_val = self.y_val.ravel()
        self.y_test = self.y_test.ravel()

        self.C = 1
        self.gamma = 0.01
        self.kernel = 'rbf'

        self.Hog_train = None
        self.Hog_test = None
        self.Hog_val = None

    def SetModel(self,kernel = None,C=None,gamma = None):
        """
        function: SetModel
        
        Sets the model atrabute based off of hyper perameter input.

        args:
            kernel (str): name of selected kernal
            C (float): value for C hyper perameter
            gamma (float): value for gamma hyper perameter

        Example:
            SVM().SetModel()
        """
            
        if kernel:
            self.kernel = kernel
        if C:
            self.C = C
        if gamma:
            self.gamma

        SVM_Model = svm.SVC(kernel=self.kernel, C=self.C, gamma = self.gamma,verbose=1)
        self.Model = SVM_Model
    
    def HOG(self,image, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(3, 3)):
        """
        function: HOG
        
        Calculates histogram of orientation for a single imae
        Steps taken are based off of a simplifed version of Scikt learn functions
        only works for MNIST 27x27 greyscale images

        args:
            image (ArrayLike): image to be processed
            orientations (int): degree of ortientations to be calculated
            pixels_per_cell (tuple): mask of number of pixels that make up a cell
            cells_per_block (tuple): mask of number of cells that will amek up block
        return:
            norm_block(ArrayLike): normalised HOG image
        Example:
            SVM().HOG(image)
        """
        gX = ndimage.sobel(image, axis=0, mode='constant')
        gy = ndimage.sobel(image, axis=1, mode='constant')

        magnitude = np.sqrt(gX**2 + gy**2)
        orientation = np.arctan2(gy, gX) * (180 / np.pi) % 180

        orientation_hist = np.zeros((27 // pixels_per_cell[0],
                                        27 // pixels_per_cell[1],
                                        orientations))
        
        #Calculates orientation histogram
        for c_row in range(orientation_hist.shape[0]):
            for c_col in range(orientation_hist.shape[1]):
                
                cell_magnitude = magnitude[c_row*pixels_per_cell[0] : (c_row + 1)*pixels_per_cell[0], c_col*pixels_per_cell[1]:(c_col + 1)*pixels_per_cell[1]]
                cell_orientation = orientation[c_row*pixels_per_cell[0] : (c_row + 1)*pixels_per_cell[0], c_col*pixels_per_cell[1] : (c_col + 1)*pixels_per_cell[1]]
                
                for p_row in range(cell_magnitude.shape[0]):
                    for p_col in range(cell_magnitude.shape[1]):

                        #Caculates cell orientation
                        orient = cell_orientation[p_row, p_col]

                        #fixes to range between 0 and 180
                        bin = int(np.floor(orient / (180 / orientations))) % orientations

                        #Update orientation histogram
                        orientation_hist[c_row, c_col, bin] += cell_magnitude[p_row, p_col]

        #calculates cell normalisation for each block
        n_blocksx = (orientation_hist.shape[0] - cells_per_block[0]) + 1
        n_blocksy = (orientation_hist.shape[1] - cells_per_block[1]) + 1

        norm_block = np.zeros((n_blocksx, n_blocksy, cells_per_block[0], cells_per_block[1], orientations))

        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = orientation_hist[x:x + cells_per_block[0], y:y + cells_per_block[1], :]
                
                #Performs L2 norm to block
                l2_norm = np.sqrt(np.sum(block ** 2) + 1e-6)  
                norm_block[x, y, :] = block / l2_norm

        return norm_block.ravel()

    def Hog_Batch(self,dataset):
        """
        function: Hog_Batch
        
        Calculates HOG for array of images

        args:
            dataset (ArrayLike): array of images to be processed
        return:
            to_return(ArrayLike): array of normalised HOG image
        Example:
            SVM().Hog_Batch(dataset)
        """
        hog_features = []

        for image in dataset:
            feature_list = self.HOG(image)
            hog_features.append(feature_list)
        
        to_return = np.array(hog_features)
        return to_return

    def Hog_Convert(self):
        """
        function: Hog Convert
        
        Calculates Batch HOG for array for train test and validation set

        Example:
            SVM().Hog_Convert(dataset)
        """
        self.Hog_test = self.Hog_Batch(self.X_test)
        self.Hog_train = self.Hog_Batch(self.X_train)
        self.Hog_val = self.Hog_Batch(self.X_val)


    def Fit(self):
        """
        function: Fit
        
        Fits set model to the training set, provides validation
        accuracy score
        
        return:
            test_acc (float) : test accuracy score
            y_pred (ArrayLike): Array of predicted labels

        Example:
            SVM().Fit()
        """
        self.Model.fit(self.Hog_train,self.y_train)
        y_pred = self.Model.predict(self.Hog_val)
        val_acc = metrics.accuracy_score(self.y_val, y_pred)

        print("Accuracy:", val_acc)

        return val_acc, y_pred


    def Test(self):
        """
        function: Test
        
        Tests a prefit model against the test set

        return:
            test_acc (float) : test accuracy score
            y_pred (ArrayLike): Array of predicted labels

        Example:
            SVM().Test()
        """
        y_pred = self.Model.predict(self.Hog_test)
        test_acc = metrics.accuracy_score(self.y_test, y_pred)

        print("Test Accuracy:",test_acc)
        
        return test_acc, y_pred



