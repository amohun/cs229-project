import cv2
import os
import numpy as np
from tqdm import tqdm

dirname = os.path.dirname(__file__)
num_images = np.loadtxt(dirname + "/train.txt").size

for i in tqdm(np.arange(num_images)):
    cv2.imwrite(dirname + "/grey_images/gframe_" + str(i) + ".jpg", 
                cv2.cvtColor(cv2.imread(dirname + "/color_images/frame_" + str(i) + ".jpg"), cv2.COLOR_RGB2GRAY))
