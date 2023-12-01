

import numpy as np
import random
from scipy.ndimage import rotate

class PreProcessing():

    def __init__(self,Dataset=None,Lables=None) -> None:
        self.Dataset = Dataset
        self.Lables = Lables

        self.Augmented_data = []
        self.augemented_lables = []

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

    def Noise(self, mean=0, sigma=10):
        Noisy = []

        for Image in self.Dataset:
            noise = np.random.normal(mean, sigma, np.shape(Image))
            noisy_image = np.clip(Image + noise, 0, 255)
            Noisy.append(noisy_image)

        to_return = Noisy#np.array(Noisy)
        return np.array(to_return)



    def Flip(self):
        Flipped = []
        for Image in self.Dataset:
            type = random.randint(0, 2)

            if type ==0:
                flipped_image = np.flip(Image,axis=0)

            elif type == 1:
                flipped_image = np.flip(Image,axis=1)

            elif type == 2:
                flipped_image = np.flip(Image,axis=0)
                flipped_image = np.flip(flipped_image,axis=0)

            Flipped.append(flipped_image )

        to_return = Flipped#np.array(Flipped)
        return to_return
    
    def New_Data(self,Loops=1):
        Processed = []
        Labels = []
        for Loop in range(0,Loops):
            Flipped = PreProcessing(self.Dataset,self.Lables).Flip()
            Noise = PreProcessing(Flipped,self.Lables).Noise()
            Rotated = PreProcessing(Noise,self.Lables).Rotate()

            Processed = Processed+Rotated
            Labels += self.Lables

        to_return = np.array(Processed).reshape((len(Processed), 28, 28, 1))

        return to_return#, Labels

    def Normalisation(Dataset):

        normalized_images = Dataset.astype('float32') / 255.0

        return normalized_images
