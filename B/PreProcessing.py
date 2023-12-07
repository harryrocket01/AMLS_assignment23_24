
import numpy as np
import random
from scipy.ndimage import rotate
from keras.utils import to_categorical

class PreProcessing():

    def __init__(self,Dataset=None,Lables=None) -> None:
        self.Dataset = Dataset
        self.Lables = Lables

        self.Augmented_data = []
        self.augemented_lables = []

        self.randomstate=42
        random.seed(self.randomstate)

    def Rotate(self):

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
        print(Rotated_Stack.shape)
        return Rotated_Stack

    def Noise(self, mean=0, sigma=5):
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
        Processed = np.empty([0, 28, 28, 3])
        Labels = []
        for Loop in range(0,Loops):
            Flipped = PreProcessing(self.Dataset,self.Lables).Flip()
            Noise = PreProcessing(Flipped,self.Lables).Noise()
            Rotated = PreProcessing(Noise,self.Lables).Rotate()

            Processed = np.concatenate((Processed, Rotated), axis=0)
            Labels += self.Lables

        to_return = np.array(Processed).reshape((len(Processed), 28, 28, 3))

        return to_return
    def normalisation(Dataset):

        normalized_images = Dataset.astype('float32') / 255.0

        return normalized_images

    def one_hot_encode(train,val,test):
        one_hot_train = to_categorical(train, num_classes=10)
        one_hot_val = to_categorical(val, num_classes=10) 
        one_hot_test = to_categorical(test, num_classes=10) 


        return np.array(one_hot_train),np.array(one_hot_val),np.array(one_hot_test)
