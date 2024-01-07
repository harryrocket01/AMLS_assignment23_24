import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import math
import random
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

    def Data_Represenation(self,x: ArrayLike, y:ArrayLike,title:str = ""):


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
    
    
    def metrics(self, true_labels:ArrayLike, pred_labels:ArrayLike):
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

