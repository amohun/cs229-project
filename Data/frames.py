import numpy as np
import cv2
import os
from tqdm import tqdm

labels = np.loadtxt(os.path.dirname(__file__) + "/train.txt")

# all frames are labeled
vidcap = cv2.VideoCapture(os.path.dirname(__file__) + "/train.mp4")
for i in tqdm(np.arange(labels.size)):
    read, image = vidcap.read()
    cv2.imwrite(os.path.dirname(__file__) + "/images/frame_%i.jpg" % i, image)
