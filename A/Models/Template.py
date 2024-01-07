

class ML_Template():
    """
    class: ML_Template

    General Template class used to make models

    Attributes:
        X_train : Array of train images
        y_train : Array of train labels
        X_val : Array of validation images
        y_val : Array of validation labels
        X_test : Array of test images
        y_test : Array of test labels

    Methods:
        Model(): Function Placeholder
        Fit(): Function Placeholder
        Train(): Function Placeholder
        Test(): Function Placeholder
    Example:
        task_instance = ML_Template()
    """
    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.Model = None

    def Model(self):
        return NotImplementedError

    def Fit(self):
        return NotImplementedError

    def Train(self):
        return NotImplementedError
    
    def Test(self):
        return NotImplementedError
