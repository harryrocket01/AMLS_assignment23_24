

import numpy as np
import random
from scipy.ndimage import rotate

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
    def __init__(self,Dataset=None,Labels=None) -> None:
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

        Randomly roatates the provided datasetm, filling the background
        with the average colour

        Example:
            roated = PreProcessing(Dataset, Labels).rotate()
        """
        Rotated = []
        back_patch = (20, 20)

        for Image in self.Dataset:
            angle = random.randint(0, 360)

            bg_color = np.mean(Image[:back_patch[0], :back_patch[1]])
            rotated_image = rotate(Image, angle, reshape=False)
            mask = rotated_image <= 0
            rotated_image[mask] = bg_color
            Rotated.append(rotated_image)
        
        to_return = Rotated#np.array(Rotated).reshape((len(Rotated), 28, 28, 1))
        return to_return

    def noise(self, mean=0, sigma=5):
        """
        function: noise

        Randomly adds gaussian addataive noise to provided images

        args :
        mean (int): mean value of the GAN
        sigma (int): divation of the GAN

        Example:
            task_instance = PreProcessing(Dataset, Labels).noise()
        """
        Noisy = []

        for Image in self.Dataset:
            noise = np.random.normal(mean, sigma, np.shape(Image))
            noisy_image = np.clip(Image + noise, 0, 255)
            Noisy.append(noisy_image)

        to_return = Noisy#np.array(Noisy)
        return np.array(to_return)

    def flip(self):
        flipped = []
        for Image in self.Dataset:
            type = random.randint(0, 2)

            if type ==0:
                flipped_image = np.flip(Image,axis=0)

            elif type == 1:
                flipped_image = np.flip(Image,axis=1)

            elif type == 2:
                flipped_image = np.flip(Image,axis=0)
                flipped_image = np.flip(flipped_image,axis=1)

            flipped.append(flipped_image )

        to_return = flipped#np.array(Flipped)
        return to_return
    
    def new_Data(self,Loops=1):
        Processed = self.Dataset.copy()
        Labels = self.Labels.copy()
        for Loop in range(0,Loops):
            Flipped = PreProcessing(self.Dataset,self.Labels).flip()
            Noise = PreProcessing(Flipped,self.Labels).noise()
            Rotated = PreProcessing(Noise,self.Labels).rotate()

            Rotated = np.array(Rotated).reshape((len(Rotated), 28, 28, 1))
            Processed = np.concatenate((Processed, Rotated), axis=0)
            Labels = np.concatenate((Labels, self.Labels), axis=0)
        return Processed, Labels

    def normalisation(Dataset):

        normalized_images = Dataset.astype('float32') / 255.0

        mean = np.mean(normalized_images, axis=(0, 1, 2))
        std = np.std(normalized_images, axis=(0, 1, 2))

        normalized_images = (normalized_images - mean) / std
        
        return normalized_images
