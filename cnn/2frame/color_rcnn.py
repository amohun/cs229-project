import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from tqdm import tqdm
import random
import matplotlib.pyplot as plt # debug

# -------------------- META DATA --------------------
cs_project = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # project root

# image size
im_height = 480
im_width = 640

# training 
batch_size = 32
test_train_split = 0.1

# model parameters
num_ip = 1
num_conv = 1
rgb_channels = 3

# -------------------- HELPER FUNCTIONS --------------------

def load_labels(project_dir):
    labels = torch.tensor(np.loadtxt(os.path.join(project_dir, "Data/train.txt"))[:, None], dtype = torch.float)
    return labels

def split_data(ip, split, n):
    train, test = np.arange(ip - 1, int(n-n*split)), np.arange(int(n-n*split), n)
    random.shuffle(train); random.shuffle(test) # shuffle in-place
    train_id = train if train.size % batch_size == 0 else train[:-(train.size % batch_size)] # size multiple of bs
    test_id = test

    np.savetxt("./train.txt", train_id)
    np.savetxt("./test.txt", test_id)

    return train_id, test_id

labels = load_labels(cs_project)
n = labels.shape[0]

train_id, test_id = split_data(num_ip, test_train_split, n) # split and shuffle data

# -------------------- MODEL DEFINITION --------------------

class cnn(nn.Module):
    def __init__(self, input_channels, num_conv):
        self.ip = input_channels
        self.num_conv = num_conv

        super().__init__()
        self.conv1 = [nn.Conv2d(rgb_channels*self.ip, rgb_channels*self.ip, padding = 2, kernel_size= (5, 5)) for _ in np.arange(num_conv)] # list of conv layers
        self.conv2 = nn.Conv2d(rgb_channels*self.ip, self.ip, padding = 2, kernel_size= (5, 5))
        self.conv3 = [nn.Conv2d(self.ip, self.ip, padding = 2, kernel_size= (5, 5)) for _ in np.arange(num_conv)]
        self.tanh = nn.Tanh()

        self.pool = nn.MaxPool2d((2, 2))
        self.flat = nn.Flatten()
        self.flin = nn.Linear(320*240, 1) # --> speed output


    def forward(self, x):
        for i in np.arange(self.num_conv):
            x = self.tanh(self.conv1[i](x))
        x = self.tanh(self.conv2(x))
        for i in np.arange(self.num_conv):
            x = self.tanh(self.conv3[i](x))
        
        x = self.pool(x)
        x = self.flat(x)
        x = self.flin(x)

        return x

def train(opt, model, batch):
    inputs = np.zeros((batch_size, rgb_channels*num_ip, im_height, im_width))

    for batch_idx, image_num in enumerate(batch): # loop through each batch element per batch
        for i in np.arange(num_ip): # build time history input
            for c in np.arange(rgb_channels):
                inputs[batch_idx, i*rgb_channels + c, :, :] = cv2.imread(cs_project + "/Data/color_images/frame_" + str(image_num - i) + ".jpg")[:, :, c]
        
        images = torch.tensor(inputs / np.max(inputs), dtype = torch.float) # normalize inputs [0, 1], convert to tensor
        y_pred = model(images)
        loss = loss_fn(y_pred, labels[batch])

        opt.zero_grad()
        loss.backward()
        opt.step()

        return opt, model, np.round(loss.item(), 2)

# -------------------- MODEL TRAINING --------------------
model = cnn(num_ip, num_conv)
loss_fn = nn.MSELoss()

# Learning rate schedule
lr_start = 2e-5 # start [twice intended start]
lr_decay = 0.7 # decay rate
num_decay = 4
lr_sch = iter(lr_start*np.cumprod(np.repeat([lr_decay], repeats = num_decay, axis = -1))) # decay schedule


n_epochs = 100
for epoch in np.arange(n_epochs):
    if epoch % (n_epochs / num_decay) == 0:
        lr = next(lr_sch)
        opt = optim.SGD(model.parameters(), lr = lr, momentum=0.9) # reinitialize mommentum 

    random.shuffle(train_id) # randomly shuffle training set
    for batch_num, batch in enumerate(tqdm(train_id.reshape(-1, batch_size))): # loop through each batch per epoch
        opt, model, loss = train(opt, model, batch)

        if batch_num % 20 == 0:
            print("Training loss: ", loss)

torch.save(model.state_dict(), "./cnn_model.pth")



