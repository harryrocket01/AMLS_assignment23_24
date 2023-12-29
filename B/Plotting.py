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
    