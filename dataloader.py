import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch

# KITTI grayscale only
class KITTI_dataset_handler():
    def __init__(self, sequence):
        # Set file paths
        self.sequence       = './dataset/sequences/{}/'.format(sequence)

        # Iteration through the names of files
        self.left_imgs      = os.listdir(self.sequence + 'image_0/')
        self.right_imgs     = os.listdir(self.sequence + 'image_1/')
        num_frames          = len(self.left_imgs)

        # Obtain calibration details 
        calibration         = pd.read_csv(self.sequence + 'calib.txt', 
                                            delimiter   = '', 
                                            header      = None, 
                                            index_col   = 0)
        self.P0             = np.array(calibration.loc['P0:']).reshape((3,4))
        self.P1             = np.array(calibration.loc['P1:']).reshape((3,4))

        # Obtain times
        self.times          = np.array(pd.read_csv(self.sequence + 'times.txt',
                                                    delimiter   = ' ',
                                                    header      = None ))

        # Load images
        self.first_img_left   = cv2.imread(self.sequence + 'image_0/'
                                            + self.left_imgs[0], 0)
        self.first_img_right  = cv2.imread(self.sequence + 'image_1/'
                                            + self.right_imgs[0], 0)
        self.second_img_left  = cv2.imread(self.sequence + 'image_0'
                                            + self.left_imgs[1], 0)
        self.second_img_right = cv2.imread(self.sequence + 'image_1/'
                                            + self.right_imgs[1], 0)

        


