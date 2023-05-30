import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt # debug

# meta
dirname = os.path.dirname(__file__)
cs_project = os.path.dirname(dirname)
test_train_split = 0.1
batch_size = 32
im_height = 480
im_width = 640
rgb_channels = 3

# load labels
labels = torch.tensor(np.loadtxt(os.path.join(dirname, "../Data/train.txt"))[:, None], dtype = torch.float)
n = labels.shape[0]

# shuffle and split data
train, test = torch.utils.data.random_split(np.arange(n), [int((1-test_train_split)*n), int(test_train_split*n)])
assert test.indices not in train.indices # disjoint sets
train_id = train.indices if len(train.indices) % batch_size == 0 else train.indices[:-(len(train.indices) % batch_size)] # size multiple of bs
test_id = test.indices

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

model = cnn()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-4, momentum = 0.9)

n_epochs = 1
inputs = np.zeros((batch_size, rgb_channels, im_height, im_width))
for _ in np.arange(n_epochs):
    for batch_num, batch in enumerate(tqdm(np.array(train_id).reshape(-1, batch_size))):
        for i, im in enumerate(batch):
            for c in np.arange(rgb_channels):
                inputs[i, c, :, :] = cv2.imread(cs_project + "/Data/images/frame_" + str(im) + ".jpg")[:, :, c]

        images = torch.tensor(inputs / np.max(inputs), dtype = torch.float) # normalize inputs [0, 1]
        y_pred = model(images)
        loss = loss_fn(y_pred, labels[batch])

        if batch_num % 20 == 0:
            print("Training loss: ", np.round(loss.item(), 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

torch.save(model.state_dict(), "./cnn_model.pth")



