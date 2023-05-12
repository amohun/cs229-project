import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# meta
dirname = os.path.dirname(__file__)
test_train_split = 0.1
batch_size = 32
im_height = 480
im_width = 640
rgb_channels = 3

# load labels
labels = np.loadtxt(os.path.join(dirname, "../Data/train.txt"))
n = labels.size

# shuffle and split data
train, test = torch.utils.data.random_split(np.arange(n), [int((1-test_train_split)*n), int(test_train_split*n)])
assert test.indices not in train.indices # disjoint sets
train_id = train.indices
test_id = test.indices

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(rgb_channels, 4*rgb_channels, kernel_size=(5, 5)) # (N, Cin, H, W)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = (2,2))


        self.conv2 = nn.Conv2d(4*rgb_channels, 4*rgb_channels, kernel_size = (5, 5))
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = (2,2))

        self.conv3 = nn.Conv2d(4*rgb_channels, rgb_channels, kernel_size = (6, 6))
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = (2,2))

        self.flat = nn.Flatten()

        self.lin4 = nn.Linear(4256*rgb_channels, 512)

        self.lin5 = nn.Linear(512, 1)

    def forward(self, x):
        # shape: N x 3 x 480 x 640
        x = self.act1(self.conv1(x))
        x = self.pool1(x)

        # shape: N x 12 x 238 x 318
        x = self.act2(self.conv2(x))
        x = self.pool2(x)

        # shape: N x 12 x 117 x 157
        x = self.act3(self.conv3(x))
        x = self.pool3(x) # N x 3 x 56 x 76

        x = self.flat(x) # N x 12768
        x = self.lin4(x) # N x 512
        x = self.lin5(x) # N x 1 --> final prediction
        return x

model = cnn()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

n_epochs = 1
for _ in np.arange(n_epochs):
    for i in np.arange(int(train_id.size / batch_size)):
        ## -----
        # load batch images
        inputs = None
        ## -----

        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

torch.save(model.state_dict(), "cnn_model.pth")



