# Borrowed from:
# https://github.com/patrickloeber/pytorchTutorial
# Visualization tool from 
# https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e

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
PATH = 'CNN_MLP_trained_model.pt'
RETRAIN = False

# Hyper-parameters 
input_size = 57600
output_size = 1
num_epochs = 50
batch_size = 300
learning_rate = 0.001
train_test_split = 0.9


if LOAD:
    # Load video
    video_path = "Data/train.mp4"
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
data = torch.load('Data/video_tensor.pt')
print(data.shape)

# Normalize data
# data = data - 128

# Load labels
labels = np.loadtxt('Data/train.txt')
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
print(len(test_loader.sampler))







# conv1 = nn.Conv2d(3, 20, 16)
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
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.conv2 = nn.Conv2d(40, 60, 5)
        self.conv3 = nn.Conv2d(60, 240, 3)
        self.resnet = nn.Linear(240, 240)  # NOT ACTUALLY RESNET
        self.fc1 = nn.Linear(240, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)
    
    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # print(x.shape)
        # x = self.drop(x)

        # Best result so far
        x = F.relu(self.resnet(x))
        x = F.relu(self.resnet(x))



        # 1 Layer Resnet did perform very well
        # x = F.relu(self.resnet(F.relu(self.resnet(x)))) + x
        # x = F.relu(self.resnet(F.relu(self.resnet(x)))) + x
        # x = self.drop(x)
        # x = F.relu(self.resnet(F.relu(self.resnet(x)))) + x
        # x = self.drop(x)
        # x = F.relu(self.resnet(F.relu(self.resnet(x)))) + x
        # x = self.drop(x)
        # x = F.relu(self.resnet(F.relu(self.resnet(x)))) + x


        x = F.relu(self.fc1(x))
        # x = self.drop(x)
        x = F.relu(self.fc2(x))
        # x = self.drop(x)
        x = self.fc3(x)
        # no activation and no softmax at the end
        return x

if RETRAIN:
    model = NeuralNet(input_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  
    losss = 0

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
            losss += loss.item() * numb
            if (i+1) % 10 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        print(f'Average MSE Loss = {losss/len(train_loader.sampler)}')
        losss = 0



    # Save the model
    torch.save(model.state_dict(), PATH)

else:
    device = torch.device('cpu')
    model = NeuralNet(57600, 1)
    model.load_state_dict(torch.load('/home/jiawenb/CS229/cs229-project/CNN_MLP_trained_model_3.9089RMSE.pt', map_location='cpu'))
    model.eval()
    criterion = nn.MSELoss()


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
        loss += delloss.item()*images.size(0)
    

    loss /= len(test_loader.sampler)
    print(f'MSE on the test data is: {loss}')
    loss = np.sqrt(loss)
    print(f'RMSE on the test data is: {loss}')

# test MSE 3.57, 3.37 best results

import matplotlib.pyplot as plt
PRPath = 'pred_real.png'
RMSEPath = 'RMSE.png'
Timeseries = 'Mes_v_time.png'
loss = 0
with torch.no_grad():
    all_outputs = []
    all_labels = []
    all_error = []
    loss = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)
        outputs = model(images)

        all_outputs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_error.extend(abs(outputs.cpu().numpy()-labels.cpu().numpy())/labels.cpu().numpy())

        # Calculate RMSE
        delloss = criterion(outputs, labels)
        loss += delloss.item() * images.size(0)
    plt.figure()
    plt.scatter(all_outputs, all_labels, s=3, c="red")
    plt.xlabel('Predicted Velocity (mph)')
    plt.ylabel('Velocity Label (mph)')
    
    plt.savefig(PRPath)
    plt.show()

    plt.figure()
    plt.scatter(all_labels, all_error, s=3, c="red")
    plt.xlabel('Velocity (mph)')
    plt.ylabel('Absolute Error(mph)')
    
    plt.savefig(RMSEPath)
    plt.show()

    plt.figure()
    timet = np.linspace(0, len(all_outputs) / 20, len(all_outputs))
    plt.plot(timet, all_outputs, color="C0")
    plt.plot(timet, all_labels, color="black")
    plt.legend(["Predicted Velocity", "Actual Velocity"])
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (mph)')
    
    plt.savefig(Timeseries)
    plt.show()
    
    loss /= len(test_loader.sampler)
    print(f'MSE on the test data is: {loss}')
    loss = np.sqrt(loss)
    print(f'RMSE on the test data is: {loss}')