"""
AMLS 1 Final Assessment - PreProcessing.py
Binary and Multivariate Classification
Author - Harry Softley-Graham
Written - Nov 2023 - Jan 2024
"""

import numpy as np
import random
from scipy.ndimage import rotate
from keras.utils import to_categorical

class PreProcessing():
    """
    class: PreProcessing

    Utility class for preprocessing data in task B

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
        task_instance = PreProcessing(Dataset, Labels)
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

        self.Augmented_data = []
        self.augemented_labels = []

        self.randomstate=42
        random.seed(self.randomstate)

    def Rotate(self):
        """
        function: rotate

        Randomly roatates the provided dataset, filling the background
        with the average colour

        return:
            to_return (ArrayLike): Array of rotated images

        Example:
            rotated = PreProcessing(Dataset, Labels).rotate()
        """
        num_samples = self.Dataset.shape[0]

        red_processed = []
        green_processed = []
        blue_processed = []

        back_patch = (20, 20)
        for sample in range(num_samples):
            angle = random.randint(0, 360)

            # Rotating the entire RGB image
            rotated_image = rotate(self.Dataset[sample], angle, reshape=False)

            # Extracting rotated color channels
            rotated_red = rotated_image[:, :, 0]
            rotated_green = rotated_image[:, :, 1]
            rotated_blue = rotated_image[:, :, 2]

            # Applying background correction
            bg_color_red = np.mean(rotated_red[:back_patch[0], :back_patch[1]])
            rotated_red[rotated_red <= 0] = bg_color_red
            red_processed.append(rotated_red)

            bg_color_green = np.mean(rotated_green[:back_patch[0], :back_patch[1]])
            rotated_green[rotated_green <= 0] = bg_color_green
            green_processed.append(rotated_green)

            bg_color_blue = np.mean(rotated_blue[:back_patch[0], :back_patch[1]])
            rotated_blue[rotated_blue <= 0] = bg_color_blue
            blue_processed.append(rotated_blue)

        Rotated_Stack = np.stack([np.array(red_processed), np.array(green_processed), np.array(blue_processed)], axis=-1)
        return Rotated_Stack

    def Noise(self, mean=0, sigma=5):
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
        Noisy = []

        for Image in self.Dataset:
                    # Generate separate noise for each channel
                    noise_r = np.random.normal(mean, sigma, Image[:, :, 0].shape)
                    noise_g = np.random.normal(mean, sigma, Image[:, :, 1].shape)
                    noise_b = np.random.normal(mean, sigma, Image[:, :, 2].shape)

                    # Add noise to each channel independently
                    noisy_image = np.stack([
                        np.clip(Image[:, :, 0] + noise_r, 0, 255),
                        np.clip(Image[:, :, 1] + noise_g, 0, 255),
                        np.clip(Image[:, :, 2] + noise_b, 0, 255)], axis=-1)

                    Noisy.append(noisy_image)

        to_return = np.array(Noisy)
        return np.array(to_return)



    def Flip(self):
        """
        function: flip

        Randomly flips images upon both axis

        return:
            to_return (ArrayLike): Array of flipped images

        Example:
            flip = PreProcessing(Dataset, Labels).rotate()
        """
        Flipped = []
        for Image in self.Dataset:
            type = random.randint(0, 2)

            if type ==0:
                flipped_image = np.flip(Image,axis=0)

            elif type == 1:
                flipped_image = np.flip(Image,axis=1)

            elif type == 2:
                flipped_image = np.flip(Image,axis=0)
                flipped_image = np.flip(flipped_image,axis=1)

            Flipped.append(flipped_image )

        to_return = np.array(Flipped)
        return to_return
    
    def data_augmentation(self,Loops=1):
        """ 
        function: data_augmentation

        augmentation data from provided dataset

        args:
            loops: number of times the dataset should be expanded

        return:
            processed(ArrayLike): array of new data
            labels (ArrayLike): array of labels for Processed

        Example:
            data = PreProcessing(Dataset, Labels).data_augmentation(3)
        """
        Processed = np.empty([0, 28, 28, 3])
        Labels = np.empty([0, 9])
        for Loop in range(0,Loops):
            Flipped = PreProcessing(self.Dataset,self.Labels).Flip()
            Noise = PreProcessing(Flipped,self.Labels).Noise()
            Rotated = PreProcessing(Noise,self.Labels).Rotate()

            Processed = np.concatenate((Processed, Rotated), axis=0)
            Labels = np.concatenate((Labels, self.Labels), axis=0)

        to_return = np.array(Processed).reshape((len(Processed), 28, 28, 3))

        return to_return
    def normalisation(Dataset):
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

        return normalised_images

    def one_hot_encode(train,val,test):
        """ 
        function: one_hot_encode

        One hot encodes labels, to be used within training the model

        args:
             train (ArrayLike): Array of labels to be one hot encoded
             val (ArrayLike): Array of images to be normalised
             test (ArrayLike): Array of images to be normalised

        Example:
            one_hot_encode = one_hot_encode(Dataset, Labels).one_hot_encode(train,val,test)
        """
        one_hot_train = to_categorical(train, num_classes=9)
        one_hot_val = to_categorical(val, num_classes=9) 
        one_hot_test = to_categorical(test, num_classes=9) 

        one_hot_train = one_hot_train.reshape((-1, 9))
        one_hot_val = one_hot_val.reshape((-1, 9))
        one_hot_test = one_hot_test.reshape((-1, 9))


        return np.array(one_hot_train),np.array(one_hot_val),np.array(one_hot_test)
