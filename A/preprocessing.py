

import numpy as np
import random
from scipy.ndimage import rotate
from numpy.typing import ArrayLike

class PreProcessing():
    """
    class: PreProcessing

    Utility class for preprocessing data in task A

    Attributes:
        Dataset (numpy.ndarray): dataset to be processed
        Labels (numpy.ndarray): labels for given dataset
        randomstate (numpy.ndarray): set random state for RNG

    Methods:
        __init__(): Initializes the class
        Rotate(): Rotates Dataset randomly
        Noise(): adds Gaussian Addative noise to images
        Flip(): randomly flips data
        New_Data(): expands dataset, applying augmentation
        Normalisation(): normalises images within dataset
    Example:
        task_instance = TaskA(Dataset, Labels)
    """
    def __init__(self, Dataset=None, Labels=None) -> None:
        """
        __init__
        Sets datasets to be processed

        args:
                Dataset (Arraylike): Array of images to be processed
                Labels (Arraylike): Array of labels for given images
        Example:
            task_instance = PreProcessing(Dataset, Labels)

        """
        self.Dataset = Dataset
        self.Labels = Labels

        self.randomstate=42
        random.seed(self.randomstate)

    def rotate(self):
        """
        function: rotate

        Randomly roatates the provided dataset, filling the background
        with the average colour

        return:
            to_return (ArrayLike): Array of rotated images

        Example:
            rotated = PreProcessing(Dataset, Labels).rotate()
        """
        to_return = []
        back_patch = (20, 20)

        for Image in self.Dataset:
            angle = random.randint(0, 360)

            bg_color = np.mean(Image[:back_patch[0], :back_patch[1]])
            rotated_image = rotate(Image, angle, reshape=False)
            mask = rotated_image <= 0
            rotated_image[mask] = bg_color
            to_return.append(rotated_image)
        
        return to_return

    def noise(self, mean: int = 0, sigma: int = 5):
        """
        function: noise

        Randomly adds gaussian addataive noise to provided images

        args :
        mean (int): mean value of the GAN
        sigma (int): divation of the GAN

        return:
            to_return (ArrayLike): Array of images with GAN

        Example:
            task_instance = PreProcessing(Dataset, Labels).noise()
        """
        to_return = []

        for Image in self.Dataset:
            noise = np.random.normal(mean, sigma, np.shape(Image))
            noisy_image = np.clip(Image + noise, 0, 255)
            to_return.append(noisy_image)

        return np.array(to_return)

    def flip(self):
        """
        function: flip

        Randomly flips images upon both axis

        return:
            to_return (ArrayLike): Array of flipped images

        Example:
            flip = PreProcessing(Dataset, Labels).rotate()
        """
        to_return = []
        for Image in self.Dataset:
            type = random.randint(0, 2)

            if type ==0:
                flipped_image = np.flip(Image,axis=0)

            elif type == 1:
                flipped_image = np.flip(Image,axis=1)

            elif type == 2:
                flipped_image = np.flip(Image,axis=0)
                flipped_image = np.flip(flipped_image,axis=1)

            to_return.append(flipped_image )

        return to_return
    
    def new_Data(self, loops: int = 1):
        """ 
        function: new_Data

        generates new data from provided dataset

        args:
            loops: number of times the dataset should be expanded

        return:
            processed(ArrayLike): array of new data
            labels (ArrayLike): array of labels for Processed

        Example:
            data = PreProcessing(Dataset, Labels).new_Data(3)
        """
        processed = self.Dataset.copy()
        labels = self.Labels.copy()
        for loop in range(0,loops):
            Flipped = PreProcessing(self.Dataset,self.Labels).flip()
            Noise = PreProcessing(Flipped,self.Labels).noise()
            Rotated = PreProcessing(Noise,self.Labels).rotate()

            Rotated = np.array(Rotated).reshape((len(Rotated), 28, 28, 1))
            processed = np.concatenate((processed, Rotated), axis=0)
            labels = np.concatenate((labels, self.Labels), axis=0)

        return processed, labels

    def normalisation(Dataset: ArrayLike):
        """ 
        function: normalisation

        normalises, centers and standardises provided images

        args:
             Dataset (ArrayLike): Array of images to be normalised

        return:
            to_return (ArrayLike): Array of normalised images

        Example:
            normalisation = PreProcessing(Dataset, Labels).normalisation()
        """
        normalised_images = Dataset.astype('float32') / 255.0

        mean = np.mean(normalised_images, axis=(0, 1, 2))
        std = np.std(normalised_images, axis=(0, 1, 2))

        to_return = (normalised_images - mean) / std
        
        return to_return
