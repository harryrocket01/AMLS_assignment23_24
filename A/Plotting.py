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

    def data_represenation(self,x: ArrayLike, y:ArrayLike, title: str):
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
    
    def confusion_matrix(self, true_labels:ArrayLike, pred_labels:ArrayLike, title: str = "")-> None:
        cm = confusion_matrix(true_labels, pred_labels)

        fig, axs = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                    xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

        axs.set_title(title)
        axs.set_xlabel("Actual Values")
        axs.set_ylabel("Predicted Values")
        
        fig.set_size_inches(3, 3)
        fig.set_tight_layout(True)

        return fig, axs

    def line_plot(self,x: ArrayLike, y:ArrayLike, title: str ="", x_label: str ="", y_label: str ="", legend: ArrayLike = []) -> (plt.Figure, plt.Axes):
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
    
    def acc_loss_plot(self, acc: ArrayLike, loss: ArrayLike, val_acc: ArrayLike, val_loss: ArrayLike,
                       x_axis, title: str):
        
        fig, axs = plt.subplots()
        axs2 = axs.twinx() 

        axs.set_xlabel('Epochs')
        axs.set_ylabel('Accuracy', color="tab:red")
        axs.plot(x_axis, acc, color="tab:red",label = 'Train Accuracy')
        axs.plot(x_axis, val_acc, color="tab:red", alpha=0.6,label = 'Val Accuracy')

        axs.tick_params(axis='y', labelcolor="tab:red")


        axs2.set_ylabel('Loss', color="tab:blue")
        axs2.plot(x_axis, loss, color="tab:blue",label = 'Train Loss')
        axs2.plot(x_axis, val_loss, color="tab:blue", alpha=0.6 ,label = 'Val Loss')
        axs2.tick_params(axis='y', labelcolor="tab:blue")

        lines, labels = axs.get_legend_handles_labels()
        lines2, labels2 = axs2.get_legend_handles_labels()
        axs2.legend(lines + lines2, labels + labels2, loc="lower right")
        
        axs.set_title(title)

        fig.set_size_inches(5.5, 3)
        fig.set_tight_layout(True)

        return fig, axs


    def learningrate_plot(self, acc1: ArrayLike, vacc1: ArrayLike, 
                          acc2: ArrayLike, vacc2: ArrayLike,
                          acc3: ArrayLike, vacc3: ArrayLike, 
                          x_axis,lr): 
        fig, axs = plt.subplots(3)

        axs[0].set_ylabel('Accuracy')
        axs[0].plot(x_axis, acc1)
        axs[0].plot(x_axis, vacc1)
        axs[0].set_title("Learning rate = "+str(lr[0]))
        axs[0].legend(["Train Accuracy","Validation Accuracy"])

        axs[1].set_ylabel('Accuracy')
        axs[1].plot(x_axis, acc2)
        axs[1].plot(x_axis, vacc2)
        axs[1].set_title("Learning rate = "+str(lr[1]))

        
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Accuracy')
        axs[2].plot(x_axis, acc3)
        axs[2].plot(x_axis, vacc3)
        axs[2].set_title("Learning rate = "+str(lr[2]))

        fig.set_tight_layout(True)
        fig.set_size_inches(5, 4)

        fig.suptitle("Effect of changing learning rate on CNN")

        return fig, axs


    def dr_bs_plot(self, acc1: ArrayLike, vacc1: ArrayLike, 
                          acc2: ArrayLike, vacc2: ArrayLike,
                          x_axis1, x_axis2): 
        
        fig, axs = plt.subplots(1, 2)

        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('BS (Log 2)')

        axs[0].plot(x_axis1, acc1)
        axs[0].plot(x_axis1, vacc1)
        axs[0].set_title("Effect of changing Batch Size")
        axs[0].set_xscale('log', base=2)

        axs[1].set_ylabel('Accuracy')
        axs[1].set_xlabel('DR')
        axs[1].plot(x_axis2, acc2)
        axs[1].plot(x_axis2, vacc2)
        axs[1].set_title("Effect of changing Dropout Rate")
        axs[1].legend(["Train Accuracy","Validation Accuracy"])

        fig.set_tight_layout(True)
        fig.set_size_inches(6, 3)

        fig.suptitle("Effect of of changing BS and DR")

        return fig, axs
    
    def metrics(self, true_labels:ArrayLike, pred_labels:ArrayLike):
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

