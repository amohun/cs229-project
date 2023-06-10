import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

import torch.nn as nn

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(rgb_channels, rgb_channels, kernel_size=(5, 5)) # (N, Cin, H, W)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(kernel_size = (2,2))

        self.conv2 = nn.Conv2d(rgb_channels, rgb_channels, kernel_size = (5, 5))
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2))

        self.conv3 = nn.Conv2d(rgb_channels, 1, kernel_size = (5, 5))
        self.act3 = nn.Tanh()
        self.flat = nn.Flatten()

        self.lin4 = nn.Linear(153*113, 512) # input feature number hand calculated from convolution
        self.lin5 = nn.Linear(512, 1)

    def forward(self, x):
        # shape: N x 3 x 480 x 640
        x = self.act1(self.conv1(x))
        x = self.pool1(x)

        # shape: N x 3 x 238 x 318
        x = self.act2(self.conv2(x))
        x = self.pool2(x)

        # shape: N x 3 x 117 x 157
        x = self.act3(self.conv3(x))
        x = self.flat(x) # --> N x 113*153

        x = self.lin4(x) # --> N x 512
        x = self.lin5(x) # --> N x 1
        return x

rgb_channels = 3
im_height = 480
im_width = 640

dirname = os.path.dirname(__file__)
cs_project = os.path.dirname(dirname)

model = cnn()
model.load_state_dict(torch.load("./cnn_model.pth"))
test = np.loadtxt("./test.txt").astype(np.int16)
labels = torch.tensor(np.loadtxt(cs_project + "/Data/train.txt")[:, None], dtype = torch.float)

loss_fn = torch.nn.MSELoss()
loss = []

image = np.zeros((rgb_channels, im_height, im_width))
for im in tqdm(test):
    for c in np.arange(rgb_channels):
        image[c, :, :] = cv2.imread(cs_project + "/Data/images/frame_" + str(im) + ".jpg")[:, :, c]

    input = torch.tensor(image / np.max(image), dtype = torch.float) # normalize inputs [0, 1]
    y_pred = model(input)
    loss.append(np.abs(y_pred.item() - labels[im].item()))

print("Mean Absolute Error: ", np.mean(loss))