import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from tqdm import tqdm
import random
import matplotlib.pyplot as plt # debug
import torch.nn.functional as F

# -------------------- META DATA --------------------
cs_project = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # project root

# image size
im_height = 480
im_width = 640

# training 
batch_size = 12
test_train_split = 0.1
n_epochs = 20


# model parameters
num_ip = 2

GPU = True

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



# Device configuration
device = torch.device('cpu')
if GPU:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class cnn(nn.Module):
    def __init__(self, input_channels):
        self.ip = input_channels

        super(cnn, self).__init__()
        # self.conv_ipip1 = nn.Conv2d(self.ip, 40, padding = 1, kernel_size= (5, 5)) # h3 igh-level conv layer
        # self.conv_ipip2 = nn.Conv2d(40, self.ip, kernel_size= (5, 5))
        # self.conv_ipip3 = nn.Conv2d(self.ip, self.ip,  kernel_size= (3, 3))
        # # self.conv_ipip4 = nn.Conv2d(self.ip, self.ip, padding = 1, kernel_size= (3, 3))

        # self.conv_ip1 = nn.Conv2d(self.ip, 1, padding = 1, kernel_size= (3, 3))
        
        # # self.conv_1 = nn.Conv2d(1, 1, padding = 1, kernel_size= (3, 3))
        # # self.conv_2 = nn.Conv2d(1, 1, padding = 1, kernel_size= (3, 3))
        # # self.conv_3 = nn.Conv2d(1, 1, padding = 1, kernel_size= (3, 3))
        # self.lin1 = nn.Linear()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.drop0p1 = nn.Dropout(0.1)
        # self.drop0p2 = nn.Dropout(0.2)

        # self.pool = nn.MaxPool2d((2, 2))
        # self.flat = nn.Flatten()
        # self.flin = nn.Linear(320*240, 1) # --> speed output

        self.conv1 = nn.Conv2d(self.ip, 10, 7)
        self.pool = nn.MaxPool2d(4, 4)
        self.drop = nn.Dropout(p=0.3, inplace=False)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 60, 3)
        self.conv4 = nn.Conv2d(60, 120, 3)
        self.resnet = nn.Linear(120, 120)  # NOT ACTUALLY RESNET
        self.fc1 = nn.Linear(120, 80)
        self.fc2 = nn.Linear(80, 60)
        self.fc3 = nn.Linear(60, 1)


    def forward(self, x):
        # x = self.drop0p1(self.tanh(self.conv_ipip1(x)))
        # x = self.drop0p2(self.tanh(self.conv_ipip2(x)))
        # x = self.drop0p2(self.relu(self.conv_ipip3(x)))

        # x = self.drop0p2(self.relu(self.conv_ip1(x)))
        
        # x = self.drop0p2(self.relu(self.conv_1(x)))
        # x = self.drop0p2(self.relu(self.conv_2(x)))
        # x = self.drop0p2(self.relu(self.conv_3(x)))

        # x = self.pool(x)
        # x = self.flat(x)
        # x = self.flin(x)


        # Jiawen Model
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)


        # Best result so far
        x = self.relu(self.resnet(x))
        x = self.relu(self.resnet(x))



        x = self.relu(self.fc1(x))
        # x = self.drop(x)
        x = self.relu(self.fc2(x))
        # x = self.drop(x)
        x = self.fc3(x)

        return x

def train(opt, model, batch):
    inputs = np.zeros((batch_size, num_ip, im_height, im_width))

    for batch_idx, image_num in enumerate(batch): # loop through each batch element per batch
        for i in np.arange(num_ip): # build time history input
            inputs[batch_idx, i, :, :] = cv2.imread(cs_project + "/Data/grey_images/gframe_" + 
                                                    str(image_num - i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        
    images = torch.tensor(inputs / np.max(inputs), dtype = torch.float).to(device) # normalize inputs [0, 1], convert to tensor
    y_pred = model(images)
    loss = loss_fn(y_pred, labels[batch].to(device))

    opt.zero_grad()
    loss.backward()
    opt.step()

    return np.round(loss.item(), 2)

# -------------------- MODEL TRAINING --------------------
model = cnn(num_ip).to(device)
loss_fn = nn.MSELoss()

# Learning rate schedule
# lr_start = 2e-8 # start [twice intended start]
# lr_decay = 0.5 # decay rate
num_decay = 1
# lr_sch = iter(lr_start*np.cumprod(np.repeat([lr_decay], repeats = num_decay, axis = -1))) # decay schedule
lr = 0.001
opt = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-3) #  momentum=0.9) # reinitialize mommentum 


inputs = np.zeros((batch_size, num_ip, im_height, im_width))
random.shuffle(train_id) # randomly shuffle training set
for epoch in np.arange(n_epochs):
    losss = 0
    # if epoch % (n_epochs / num_decay) == 0:
        # lr = next(lr_sch)
        
        # opt = optim.Adam(model.parameters(), lr = lr) #  momentum=0.9) # reinitialize mommentum 

    # random.shuffle(train_id) # randomly shuffle training set
    for batch_num, batch in enumerate(tqdm(train_id.reshape(-1, batch_size))): # loop through each batch per epoch
        loss = train(opt, model, batch)
        losss += loss
        if batch_num % 20 == 0:
            print("Training loss: ", loss)
    print("Epoch: ", epoch + 1, "Training loss: ", losss / len(train_id))

torch.save(model.state_dict(), "./cnn_model.pth")



