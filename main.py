<<<<<<< HEAD
import os
import numpy as np
import pandas as pd
import cv2
from sympy import I
import torch
import io 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import KITTI_dataset_handler

KITTI = KITTI_dataset_handler('00', transform = None) # 00-10

# number of imgs in a folder
print(len(KITTI.left_imgs))

# plotting of samples
fig = plt.figure()

for i in range(len(KITTI)):
    img_name    = os.path.join(KITTI.sequence_path, KITTI.left, KITTI.left_imgs[i])
    img_name    = cv2.imread(img_name)
    
    fig         = plt.subplot(1, 4, i + 1)
    fig.set_title('Sample #{}'.format(i))
    fig.axis('off')  
    plt.imshow(img_name)

    if i == 3:
        plt.show()
        break
=======
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


>>>>>>> 442c458437b5603e39aa3ee5c85b81737e91b7d8
