import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Plotting ():

    def __init__(self) -> None:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 8

        plt.rcParams['axes.labelsize'] = 8
        plt.rcParams['axes.titlesize'] = 8

        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8

        plt.rcParams['lines.linewidth'] = 0.7

        plt.tight_layout()

    def Data_Represenation(self,x: ArrayLike, y:ArrayLike, title: str):
        True_Example = []
        False_Example = []

        Example_size = 1

        for index in range(0, len(x)):
            if y[index] == 1 and len(True_Example) < Example_size:
                True_Example.append(x[index])
            elif y[index] == 0 and len(False_Example) < Example_size:
                False_Example.append(x[index])
            elif len(True_Example) == Example_size and len(False_Example) == Example_size:
                break

        fig, axs = plt.subplots(Example_size, 2)

        if Example_size == 1:
            axs = np.expand_dims(axs, axis=0)  # Ensure axs is 2D even for a single example

        row = 0
        for Image in True_Example:
            subplot = axs[row, 1]
            subplot.imshow(Image)
            subplot.axis("off")
            subplot.set_title("Example of 1")

            row += 1

        row = 0
        for Image in False_Example:
            subplot = axs[row, 0]
            subplot.imshow(Image)
            subplot.axis("off")
            subplot.set_title("Example of 0")
            row += 1

        fig.set_size_inches(4, 0.2+2*Example_size)
        fig.suptitle(title)
        fig.set_tight_layout(True)
        return axs, fig
    
    def Confusion_Matrix(self, true_labels:ArrayLike, pred_labels:ArrayLike, title: str = "")-> None:
        cm = confusion_matrix(true_labels, pred_labels)

        fig, axs = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                    xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
        fig.set_tight_layout(True)
        axs.set_title(title)

        return fig, axs

    def Line_Plot(self,x: ArrayLike, y:ArrayLike, title: str, x_label: str, y_label: str, legend) -> (plt.Figure, plt.Axes):
        fig, axs = plt.subplots()
        counter = 1

        if np.array(y).ndim >= 2:
            print(True)
            for Y_current in y:
                axs.plot(x, Y_current)
        else:
            axs.plot(x, y)
            

        axs.set_title(title)
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        axs.legend(legend)
        fig.set_tight_layout(True)
        fig.set_size_inches(3.5, 2.5)

        return fig, axs
    
    def acc_loss_plot(acc,loss,val_acc,val_loss,x_axis):
        fig, axs = plt.subplots()
        axs2 = axs.twinx() 

        axs.set_xlabel('Epochs')
        axs.set_ylabel('Accuracy', color="tab:red")
        axs.plot(x_axis, acc, color="tab:red")
        axs.plot(x_axis, val_acc, color="tab:red", alpha=0.8)

        axs.tick_params(axis='y', labelcolor="tab:red")


        axs2.set_ylabel('Loss', color="tab:blue")
        axs2.plot(x_axis, loss, color="tab:blue")
        axs2.plot(x_axis, val_loss, color="tab:blue", alpha=0.8)
        axs2.tick_params(axis='y', labelcolor="tab:blue")
        
        fig.set_size_inches(3.5, 2)
        fig.set_tight_layout(True)




    def HP_Line_plot(data,lr,bs,epochs):
        pass
    
    def Metrics(self, true_labels:ArrayLike, pred_labels:ArrayLike):
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        TP = np.sum((true_labels == 1) & (pred_labels == 1))
        FP = np.sum((true_labels == 0) & (pred_labels == 1))
        TN = np.sum((true_labels == 0) & (pred_labels == 0))
        FN = np.sum((true_labels == 1) & (pred_labels == 0))

        Accuracy = (TP+TN)/(TP+FP+FN+TN)

        Precision = (TP)/(TP+FP)

        Recall = (TP)/(TP+FN)

        Specificity = (TN)/(TN+FP)

        F1_score = (2*Precision*Recall)/(Precision+Recall)

        print("Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | Specificity: {:.4f} | F1_score: {:.4f}".format(
            Accuracy,Precision,Recall,Specificity,F1_score))
        
        return Accuracy,Precision,Recall,Specificity,F1_score

