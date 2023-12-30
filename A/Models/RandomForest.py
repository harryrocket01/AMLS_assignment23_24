
import sklearn

from sklearn import metrics

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Models.Template import ML_Template

class RandomForest(ML_Template):


    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)
    

        self.X_train=self.X_train.reshape(len(self.y_train), 28*28)
        self.X_val=self.X_val.reshape(len(self.y_val), 28*28)
        self.X_test=self.X_test.reshape(len(self.y_test), 28*28)

        self.y_train = self.y_train.ravel()
        self.y_val = self.y_val.ravel()
        self.y_test = self.y_test.ravel()
        self.n_estimators = 10
        self.random_state=42
  



    def SetModel(self,n_estimators = None,random_state=None):
                
        if n_estimators:
            self.n_estimators = n_estimators
        if random_state:
            self.random_state = random_state


        random_f = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        
        self.Model = random_f

    def Fit(self):
        self.Model.fit(self.X_train,self.y_train)
        y_pred = self.Model.predict(self.X_val)
        val_acc = metrics.accuracy_score(self.y_val, y_pred)

        print("Accuracy:", val_acc)

        return val_acc, y_pred


    def Test(self):
        y_pred = self.Model.predict(self.X_test)
        test_acc = metrics.accuracy_score(self.y_test, y_pred)

        print("Test Accuracy:",test_acc)
        
        return test_acc, y_pred


class ADABoost(RandomForest):


    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        super().__init__(X_train,y_train,X_val,y_val,X_test,y_test)

        self.n_estimators = 50
        self.random_state= 42
        self.ada_n_estimators = 250



    def SetModel(self,n_estimators = None,random_state=None,ada_n_estimators=None):
        
        if n_estimators:
            self.n_estimators = n_estimators
        if random_state:
            self.random_state = random_state
        if ada_n_estimators:
            self.ada_n_estimators = ada_n_estimators


        random_f = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        ada_boost = AdaBoostClassifier(estimator=random_f, n_estimators=self.ada_n_estimators, random_state=self.random_state)

        self.Model = ada_boost