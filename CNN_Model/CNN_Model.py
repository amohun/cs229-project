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

LOAD = False # If loading mp4 data from scratch
DIFF = False  # Take difference bt. frames
SHUFFLE = True  # Shuffle the train dataset
GPU = True  # Use GPU


# Hyper-parameters 
input_size = 57600
output_size = 1
num_epochs = 20
batch_size = 100
learning_rate = 0.001
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
        print(f'frame {i}')
    video_tensor = torch.stack(frames)

    # Check video tensor shape
    print(video_tensor.shape)

    # Resize video tensor


    print(video_tensor.shape)
    # Save video tensor
    torch.save(video_tensor, 'cs229-project/Data/video_tensor.pt')

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
device = torch.device('cpu')
if GPU:
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

print('loaded ds')

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
print('Example data shape: ', example_data.shape)
print('Example targets shape: ', example_targets.shape)








# conv1 = nn.Conv2d(3, 40, 5)
# a = conv1(example_data)
# print(a.shape)
# pool1 = nn.MaxPool2d(4, 4)
# a = pool1(a)
# print(a.shape)
# conv2 = nn.Conv2d(40, 60, 5)
# a = conv2(a)
# print(a.shape)
# a = pool1(a)
# print(a.shape)
# a = torch.flatten(a, 1)
# print(a.shape)











# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.conv1 = nn.Conv2d(3, 40, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(40, 60, 5)
        self.conv3 = nn.Conv2d(60, 240, 3)
        self.fc1 = nn.Linear(240, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # no activation and no softmax at the end
        return x

model = NeuralNet(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
losss = 0
sizee = 0

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        numb = labels.shape[0]
        losss += losss * numb
        sizee += numb
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    print(f'Avg Loss = {losss/sizee}')
    losss = 0
    sizee = 0

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)
        outputs = model(images)
        
        # Calculate RMSE
        delloss = criterion(outputs, labels)
        loss += delloss.item()
    

    loss /= len(test_loader)
    loss = np.sqrt(loss)
    print(f'MSE on the test data is: {loss}')