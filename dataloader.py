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
        self.sequence_path  = './dataset/sequences/{}/'.format(sequence)
        self.poses_path     = './dataset/poses/{}.txt'.format(sequence)
        poses               = pd.read_csv(self.poses_path, delimiter=' ', header=None)
        
        # Iteration through the names of files
        self.left_imgs      = os.listdir(self.sequence_path + 'image_0')
        self.right_imgs     = os.listdir(self.sequence_path + 'image_1')

        # Obtain calibration details 
        calibration         = pd.read_csv(self.sequence_path + 'calib.txt', 
                                            delimiter   = ' ', 
                                            header      = None, 
                                            index_col   = 0)
        self.P0             = np.array(calibration.loc['P0:']).reshape((3,4))
        self.P1             = np.array(calibration.loc['P1:']).reshape((3,4))

        # Obtain times and ground truth poses 
        self.times          = np.array(pd.read_csv(self.sequence_path + 'times.txt',
                                                    delimiter   = ' ',
                                                    header      = None ))
        self.gt             = np. zeros((len(poses), 3, 4))
        for i in range(len(poses)):
            self.gt[i]      = np.array(poses.iloc[i]).reshape((3, 4))

        # Load images
        self.first_img_left   = cv2.imread(self.sequence_path + 'image_0/'
                                            + self.left_imgs[0], 0)
        self.first_img_right  = cv2.imread(self.sequence_path + 'image_1/'
                                            + self.right_imgs[0], 0)
        self.second_img_left  = cv2.imread(self.sequence_path + 'image_0/'
                                            + self.left_imgs[1], 0)
        self.second_img_right = cv2.imread(self.sequence_path + 'image_1/'
                                            + self.right_imgs[1], 0)

        # height and width 
        self.imw              = self.first_img_left.shape[1]
        self.imh              = self.first_img_left.shape[0]

    def __len__(self):
        return len(self.left_imgs)
