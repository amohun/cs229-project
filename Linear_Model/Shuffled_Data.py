# Borrowed from:
# https://github.com/patrickloeber/pytorchTutorial

# Shuffled data with plotted actual vs predicted value

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision
import matplotlib.pyplot as plt
import os
import pickle as pkl
import itertools

LOAD = False

# Hyper-parameters
input_size = 57600
output_size = 1
num_epochs = 1000
batch_size = 30
learning_rate = 0.0000000001
train_test_split = 0.8

if LOAD:
    # Load video
    video_path = "Data/train.mp4"
    stream = "video"
    video = torchvision.io.VideoReader(video_path, 'video')
    print(video.get_metadata())

    # Convert video to tensor
    frames = []
    for i, frame in enumerate(video):  # itertools.takewhile(lambda x: x['pts'] <= 5, video.seek(2))):
        frames.append(F2.resize(frame['data'], size=[120, 160], antialias=False))
        print(f'frame {i}')
    video_tensor = torch.stack(frames)

    # Check video tensor shape
    print(video_tensor.shape)

    # Resize video tensor

    print(video_tensor.shape)
    # Save video tensor
    torch.save(video_tensor, 'Data/video_tensor.pt')

# Load video tensor
data = torch.load('Data/video_tensor.pt')
print(data.shape)

# Load labels
labels = np.loadtxt('Data/train.txt')
labels = torch.tensor(labels)
print(labels.shape)

# Only use first 10000 frames
# data = data[:10000]
# labels = labels[:10000]

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create dataset
class Speed_Dataset():

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.n_samples = data.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = data.type(torch.FloatTensor)  # size [n_samples, n_features]
        self.y_data = labels.type(torch.FloatTensor)  # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = Speed_Dataset()

# create an instance of the Speed_Dataset class
dataset = Speed_Dataset()

# get the number of samples in the dataset
n_samples = len(dataset)

# generate a random permutation of the indices of the dataset
perm = torch.randperm(n_samples)

# use the permutation to shuffle the data
shuffled_x_data = dataset.x_data[perm]
shuffled_y_data = dataset.y_data[perm]

# create a new Speed_Dataset object with the shuffled data
shuffled_dataset = Speed_Dataset()
shuffled_dataset.x_data = shuffled_x_data
shuffled_dataset.y_data = shuffled_y_data


dataset = shuffled_dataset

# Split dataset into train and test
train_size = int(train_test_split * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = (torch.utils.data.Subset(dataset, range(train_size)),
                         torch.utils.data.Subset(dataset, range(train_size, len(dataset))))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = next(examples)
print('Example data shape: ', example_data.reshape(-1, input_size).shape)
print('Example targets shape: ', example_targets.shape)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.unsqueeze(1).to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

loss = 0
var = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.unsqueeze(1).to(device)
        outputs = model(images)
        print(labels)
        plt.scatter(outputs.numpy(),labels.numpy())
        plt.xlabel('Predicted Velocity (mph)')
        plt.ylabel('Velocity Label (mph)')
        # Calculate RMSE
        delloss = criterion(outputs, labels)
        loss += delloss.item()
        var = var + torch.var(labels)
    plt.show()
    #print(var)
    loss /= len(test_loader)
    loss = np.sqrt(loss)
    print(f'MSE on the test data is: {loss}')
