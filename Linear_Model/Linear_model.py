# Borrowed from:
# https://github.com/patrickloeber/pytorchTutorial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision
import pickle as pkl
import itertools

LOAD = False  # If loading mp4 data from scratch
DIFF = False  # Take difference bt. frames
SHUFFLE = True  # Shuffle the train data


# Hyper-parameters 
input_size = 57600
output_size = 1
num_epochs = 500
batch_size = 30
learning_rate = 0.000000001
train_test_split = 0.8


if LOAD:
    # Load video
    video_path = "cs229-project/Data/train.mp4"
    stream = "video"
    video = torchvision.io.VideoReader(video_path, 'video')
    print(video.get_metadata())

    # Convert video to tensor
    frames = []
    for i, frame in enumerate(video):  #itertools.takewhile(lambda x: x['pts'] <= 5, video.seek(2))):
        frames.append(F2.resize(frame['data'], size = [120, 160], antialias=False))
        # print(frame['data'].shape)
        print(f'frame {i}')
    video_tensor = torch.stack(frames).type(torch.uint8)

    # Check video tensor shape
    print(video_tensor.shape)

    # Resize video tensor


    # Save video tensor
    torch.save(video_tensor, 'cs229-project/Data/video_tensor.pt')

    print('saved')


# Load video tensor
data = torch.load('cs229-project/Data/video_tensor.pt')
print(data.shape)

# Load labels
labels = np.loadtxt('cs229-project/Data/train.txt')
labels = torch.tensor(labels)
print(labels.shape)

if DIFF:
    # Take difference between frames
    data = data[1:] - data[:-1]
    labels = labels[1:]
    print(data.shape)
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
        self.x_data = data.type(torch.FloatTensor) # size [n_samples, n_features]
        self.y_data = labels.type(torch.FloatTensor) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = Speed_Dataset()

# Split dataset into train and test
train_size = int(train_test_split * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = (torch.utils.data.Subset(dataset, range(train_size)), 
                         torch.utils.data.Subset(dataset, range(train_size, len(dataset))))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=batch_size, 
                                           shuffle=SHUFFLE)

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


losss = 0
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
        
        losss += loss.item() * images.shape[0]
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    print(f'Average Loss = {losss/len(train_loader.sampler)}')
    losss = 0


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.unsqueeze(1).to(device)
        outputs = model(images)
        
        # Calculate RMSE
        delloss = criterion(outputs, labels)
        loss += delloss.item() * images.shape[0]
    

    loss /= len(test_loader.sampler)
    loss = np.sqrt(loss)
    print(f'MSE on the test data is: {loss}')