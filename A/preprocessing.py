

import numpy as np
import random
from scipy.ndimage import rotate

class PreProcessing():

    def __init__(self,Dataset=None,Labels=None) -> None:
        self.Dataset = Dataset
        self.Labels = Labels

        self.Augmented_data = []
        self.augemented_Labels = []

        self.randomstate=42
        random.seed(self.randomstate)

    def Rotate(self):
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

    def Noise(self, mean=0, sigma=5):
        Noisy = []

        for Image in self.Dataset:
            noise = np.random.normal(mean, sigma, np.shape(Image))
            noisy_image = np.clip(Image + noise, 0, 255)
            Noisy.append(noisy_image)

        to_return = Noisy#np.array(Noisy)
        return np.array(to_return)



    def Flip(self):
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
    
    def New_Data(self,Loops=1):
        Processed = self.Dataset.copy()
        Labels = self.Labels.copy()
        for Loop in range(0,Loops):
            Flipped = PreProcessing(self.Dataset,self.Labels).Flip()
            Noise = PreProcessing(Flipped,self.Labels).Noise()
            Rotated = PreProcessing(Noise,self.Labels).Rotate()

            Rotated = np.array(Rotated).reshape((len(Rotated), 28, 28, 1))
            Processed = np.concatenate((Processed, Rotated), axis=0)
            Labels = np.concatenate((Labels, self.Labels), axis=0)
        return Processed, Labels

    def Normalisation(Dataset):

        normalized_images = Dataset.astype('float32') / 255.0

        mean = np.mean(normalized_images, axis=(0, 1, 2))
        std = np.std(normalized_images, axis=(0, 1, 2))

        normalized_images = (normalized_images - mean) / std
        
        return normalized_images
