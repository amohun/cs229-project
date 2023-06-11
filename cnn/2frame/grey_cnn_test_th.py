import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, input_channels):
        self.ip = input_channels

        super().__init__()
        self.conv_ipip1 = nn.Conv2d(self.ip, self.ip, padding = 1, kernel_size= (3, 3)) # h3 igh-level conv layer
        self.conv_ipip2 = nn.Conv2d(self.ip, self.ip, padding = 1, kernel_size= (3, 3))
        self.conv_ipip3 = nn.Conv2d(self.ip, self.ip, padding = 1, kernel_size= (3, 3))
        self.conv_ip1 = nn.Conv2d(self.ip, 1, padding = 1, kernel_size= (3, 3))
        self.conv_1 = nn.Conv2d(1, 1, padding = 1, kernel_size= (3, 3))
        self.conv_2 = nn.Conv2d(1, 1, padding = 1, kernel_size= (3, 3))

        self.tanh = nn.Tanh()

        self.pool = nn.MaxPool2d((2, 2))
        self.flat = nn.Flatten()
        self.flin = nn.Linear(320*240, 1) # --> speed output


    def forward(self, x):
        x = self.tanh(self.conv_ipip1(x))
        x = self.tanh(self.conv_ipip2(x))
        x = self.tanh(self.conv_ipip3(x))

        x = self.tanh(self.conv_ip1(x))
        
        x = self.tanh(self.conv_1(x))
        x = self.tanh(self.conv_2(x))

        x = self.pool(x)
        x = self.flat(x)
        x = self.flin(x)

        return x

num_ip = 2
im_height = 480
im_width = 640

dirname = os.path.dirname(os.path.dirname(__file__))
cs_project = os.path.dirname(dirname)

model = cnn(num_ip)
model.load_state_dict(torch.load("./sat_night.pth"))
test = np.loadtxt("./test.txt").astype(np.int16)
labels = torch.tensor(np.loadtxt(cs_project + "/Data/train.txt")[:, None], dtype = torch.float)

loss_fn = torch.nn.MSELoss()
loss = np.zeros(test.size,)

image = np.zeros((num_ip, im_height, im_width))
for iter, im in tqdm(enumerate(test)):
    for i in np.arange(num_ip):
        image[i, :, :] = cv2.imread(cs_project + "/Data/grey_images/gframe_" + str(im - i) + ".jpg", cv2.IMREAD_GRAYSCALE)

    input = torch.tensor(image / np.max(image), dtype = torch.float) # normalize inputs [0, 1]
    y_pred = model(input)
    loss[iter] = y_pred.item() - labels[im].item()

print("RMS Error: ", np.sqrt(np.mean(loss**2)))