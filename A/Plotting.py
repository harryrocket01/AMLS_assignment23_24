import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

class Plotting ():

    def __init__(self) -> None:
        pass

    def Data_Represenation(x: ArrayLike, y:ArrayLike):

        True_Example = []
        False_Example = []

        Example_size = 3

        for index in range (0,len(x)):
            if y[index] ==1 and len(True_Example) <Example_size:
                True_Example.append(x[index])
            elif y[index] ==1 and len(False_Example) <Example_size:
                False_Example.append(x[index])
            elif (len(True_Example) == Example_size-1) and (len(False_Example) == Example_size-1):
                break
        
        fig, axs = plt.subplots(3,2)

        row = 0
        for Image in True_Example:
            subplot = axs[row, 0]
            subplot.imshow(Image)
            subplot.axis("off")
            subplot.set_title("Example of 1")

            row +=1

        row = 0
        for Image in False_Example:
            subplot = axs[row, 1]
            subplot.imshow(Image)
            subplot.axis("off")
            subplot.set_title("Example of 0")
            row +=1        
        
        print(len(True_Example),len(False_Example))

        plt.show()
        
        return axs, fig
    
    def Confusion_Matrix()-> None:
        pass
    
    def Line_Plot(x: ArrayLike, y:ArrayLike, title: str, x_label: str, y_label: str, legend) -> (plt.Figure, plt.Axes):
        fig, ax = plt.subplots()
        counter = 1

        if isinstance(y, np.ndarray) and y.ndim == 2:
            for Y_current in y:
                ax.plot(x, Y_current)
        else:
            ax.plot(x, y)
            

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(legend)