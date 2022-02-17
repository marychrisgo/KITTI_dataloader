import os
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import KITTI_dataset_handler

KITTI = KITTI_dataset_handler('00')

img = KITTI.left_imgs[0] # how to get the path so I can show it?

img1 = mpimg.imread('./dataset/sequences/00/image_0/' + img)
plt.imshow(img1)
plt.show()


