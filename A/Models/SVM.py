import numpy as np
from numpy.typing import ArrayLike

import pandas as pd

import sklearn
from sklearn.svm import SVC
from sklearn import svm

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from skimage import color
from skimage.feature import hog
from scipy import ndimage

from sklearn.metrics import classification_report,accuracy_score


#Import ML Template
from Models.Template import ML_Template

class SVM(ML_Template):


    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)

        self.X_train=self.X_train.reshape(len(self.y_train), 28*28)
        self.X_val=self.X_val.reshape(len(self.y_val), 28*28)
        self.X_test=self.X_test.reshape(len(self.y_test), 28*28)

        self.y_train = self.y_train.ravel()
        self.y_val = self.y_val.ravel()
        self.y_test = self.y_test.ravel()

        self.C = 1
        self.gamma = 0.1
        self.kernel = 'rbf'



    def SetModel(self,kernel = None,C=None,gamma = None):
        
        if kernel:
            self.kernel = kernel
        if C:
            self.C = C
        if gamma:
            self.gamma

        SVM_Model = svm.SVC(kernel=self.kernel, C=self.C, gamma = self.gamma)
        self.Model = SVM_Model

    def Fit(self):
        self.Model.fit(self.X_train,self.y_train)


    def Test(self):
        y_pred = self.Model.predict(self.X_test)
    
        #replace with my own accuracy score
        print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))

        

    def BayOpt(self,Crange = [0.1, 1, 10, 100],gammarange = [1, 0.1, 0.01, 0.001],kernalrange =['linear', 'poly', 'rbf', 'sigmoid'] ):
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


    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)

        self.X_train=self.X_train.squeeze(axis=-1)

        self.X_val=self.X_val.squeeze(axis=-1)

        self.X_test=self.X_test.squeeze(axis=-1)


        self.y_train = self.y_train.ravel()
        self.y_val = self.y_val.ravel()
        self.y_test = self.y_test.ravel()

        self.C = 1
        self.gamma = 0.1
        self.kernel = 'rbf'

        self.Hog_train = None
        self.Hog_test = None
        self.Hog_val = None

    def SetModel(self,kernel = None,C=None,gamma = None):
            
            if kernel:
                self.kernel = kernel
            if C:
                self.C = C
            if gamma:
                self.gamma

            SVM_Model = svm.SVC(kernel=self.kernel, C=self.C, gamma = self.gamma)
            self.Model = SVM_Model


    def Model(self,Kernal = "RBF" ):
        SVM_Model = svm.SVC(kernal=Kernal)
        self.Model = SVM_Model
    
    def HOG(self,image, orientations=8, pixels_per_cell=(3, 3), cells_per_block=(3, 3)):
        #Simplified version of the process the Sklearn Image function uses, only works for MNIST 27x27 greyscale images

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

                        #Update Orientation Histogram
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
                norm_block = block / l2_norm
                norm_block[x, y, :] = norm_block

        return norm_block.ravel()

    def Hog_Batch(self,dataset):
        
        hog_features = []

        for image in dataset:
            feature_list = self.HOG(image)
            hog_features.append(feature_list)
        
        to_return = np.array(hog_features)
        return to_return

    def Hog_Convert(self):
        self.Hog_test = self.Hog_Batch(self.X_test)
        self.Hog_train = self.Hog_Batch(self.X_train)
        self.Hog_val = self.Hog_Batch(self.X_val)


    def Fit(self):
        print(self.Hog_train)
        try:
            self.Model.fit(self.Hog_train,self.y_train)
        except:
            self.Model.fit(self.Hog_train,self.y_train)
 

    def Test(self):
        try:
            y_pred = self.Model.predict(self.Hog_test)
        
            #replace with my own accuracy score
            print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))
        except:
            y_pred = self.Model.predict(self.Hog_test)
            print("Accuracy:",metrics.accuracy_score(self.y_test, y_pred))



