import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import math
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix

class Plotting ():
    """
    class: Plotting

    Plotting class used to plot data in task B

    Attributes:
        plt.rcParams: Matplot Lib instances with IEEE plotting styling

    Methods:
        __init__(): Initializes the class
        data_represenation(): Plots an example of each class
        confusion_matrix(): Plots a confusion Matrix 
        line_plot(): Plots a line plot
        acc_loss_plot(): plots accuracy loss plot
        metrics()
    Example:
        task_instance = Plotting()
    """
    def __init__(self) -> None:
        """
        __init__

        sets plotting style to IEEE for plotting

        return:
            to_return (ArrayLike): Array of rotated images

        Example:
            task_instance = Plotting()
        """

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 8

        plt.rcParams['axes.labelsize'] = 8
        plt.rcParams['axes.titlesize'] = 8

        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8

        plt.rcParams['lines.linewidth'] = 0.7

        plt.tight_layout()

    def Data_Represenation(self,x: ArrayLike, y:ArrayLike,title:str = ""):

        """
        function: data_represenation

        Creates a plot that shows and example of each of the 9 class

        args:
            x (ArrayLike): images to represent
            y (ArrayLike): labels of class
            title (str): title for plot

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).data_represenation(x,y,title)
        """
        Unique = np.unique(y)
        row = round(len(Unique)**0.5)
        columns = len(Unique)//row

        fig, axs = plt.subplots(row, columns)

        counter = 0
        for row_index in range(0, row):
            for column_index in range(0, columns):
                subplot = axs[row_index, column_index]
                image_index = np.where(y == Unique[counter])
                image_index = image_index[0][random.randint(0, len(image_index[0]))]
                subplot.imshow(x[image_index])

                subplot.axis("off")
                subplot.set_title("Example of " + str(Unique[counter]))
                counter += 1

        fig.set_size_inches(1*row, 0.2+1*columns)
        fig.suptitle(title)
        fig.set_tight_layout(True)

        return fig, axs


    def Confusion_Matrix(self, true_labels:ArrayLike, pred_labels:ArrayLike, title: str = "")-> None:
        """
        function: confusion_matrix

        Creates a confusion matrix plot from true and predicted labels

        args:
            true_labels (ArrayLike): Array of True Labels
            pred_labels (ArrayLike): Array of Predicted Labels
            title (string): Title for plot
        
        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).confusion_matrix(true_labels,pred_labels,title)
        """
        cm = confusion_matrix(true_labels, pred_labels)

        fig, axs = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                    xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
        fig.set_tight_layout(True)
        axs.set_title(title)

        axs.set_xlabel("Actual Values")
        axs.set_ylabel("Predicted Values")

        return fig, axs

    def Line_Plot(self,x: ArrayLike, y:ArrayLike, title: str, x_label: str, y_label: str, legend) -> (plt.Figure, plt.Axes):
        """
        function: line_plot

        Creates a line plot from two inputted arrays
        
        args:
            x (ArrayLike): array to plot on x axis
            y (ArrayLike): array to plot on y axis
            title (str): title for plot
            x_label (str): X axis Label
            y_label (str): Y axis Label
            legend (ArrayLike): Array for plot legend

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).line_plot(x,y,title)
        """
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
        """
        function: acc_loss_plot

        Creates a two y-axis accuracy-loss plot from inputted arrays. Used
        to show training of a model
        
        args:
            acc (ArrayLike): Array of train accuracy points
            loss (ArrayLike): Array of train loss points
            val_acc (ArrayLike): Array of validation accuracy points
            val_loss (ArrayLike): Array of validation loss points
            x_axis (ArrayLike): array of epochs
            title (str): Title for plot

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).acc_loss_plot(acc,loss,v_acc,v_loss)
        """

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
    
    def metrics(self, true_labels:ArrayLike, pred_labels:ArrayLike):
        """
        function: metrics

        Generates metrics used to evaluate models. This includes:
        accuracy,precision,recall,specificity and f1 score.
        
        args:
            true_labels (ArrayLike): Array of True Labels
            pred_labels (ArrayLike): Array of Predicted Labels

        return:
            Accuracy (float): Accuracy of the given classifcation prediction
            Precision (float): Precision of the given classifcation prediction
            Recall (float): Recall of the given classifcation prediction
            Specificity (float): Specificity of the given classifcation prediction
            F1_score (float): F1 Score of the given classifcation prediction

        Example:
            fig, axs = Plotting(Dataset, Labels).metrics(true_labels,pred_labels)
        """
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        num_classes = len(np.unique(true_labels))

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for i in range(num_classes):
            for j in range(num_classes):
                confusion_matrix[i, j] = np.sum((true_labels == i) & (pred_labels == j))

        precision = np.zeros(num_classes, dtype=float)
        recall = np.zeros(num_classes, dtype=float)
        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        for i in range(num_classes):
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i, :]) - TP

            if TP + FP != 0:
                precision[i] = TP / (TP + FP)
            else:
                precision[i] = 0
            if TP + FP != 0:
                recall[i] = TP / (TP + FN)
            else:
                recall[i] = 0

        if (np.mean(precision) + np.mean(recall)) != 0:
            f1_score = 2 * np.mean(precision) * np.mean(recall) / (np.mean(precision) + np.mean(recall)) 
        else:
            f1_score=  0

        print("Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1_score: {:.4f}".format(
            accuracy, np.mean(precision), np.mean(recall), f1_score))

        return accuracy, np.mean(precision), np.mean(recall), f1_score

