import torch
import numpy as np
import cv2
import os
from tqdm import tqdm

import torch.nn as nn


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


num_ip = 2
im_height = 480
im_width = 640

dirname = os.path.dirname(os.path.dirname(__file__))
cs_project = os.path.dirname(dirname)

model = cnn(num_ip)
model.load_state_dict(torch.load("/home/jiawenb/CS229/cs299-Matt/cs229-project/cnn/2frame/cnn_model_3.9.pth"))
model.eval()
test = np.loadtxt("./test.txt").astype(np.int16)
labels = torch.tensor(np.loadtxt(cs_project + "/Data/train.txt")[:, None], dtype = torch.float)

loss_fn = torch.nn.MSELoss()
loss = []

import matplotlib.pyplot as plt

PRPath = 'pred_real.png'
RMSEPath = 'RMSE.png'
Timeseries = 'Mes_v_time.png'


image = np.zeros((1, num_ip, im_height, im_width))

all_outputs = np.empty(test.shape)
all_labels = np.empty(test.shape)
all_error = []
with torch.no_grad():
    for im in tqdm(test):
        for i in np.arange(num_ip):
            image[0, i, :, :] = cv2.imread(cs_project + "/Data/grey_images/gframe_" + str(im - i) + ".jpg", cv2.IMREAD_GRAYSCALE)

        input = torch.tensor(image / np.max(image), dtype = torch.float) # normalize inputs [0, 1]
        # print(input.shape)
        y_pred = model(input)
        all_outputs[im-18360] = (y_pred.item())
        all_labels[im-18360] = (labels[im].item())
        # all_error.extend(abs(y_pred.item()-labels[im].item())/labels[im].item())
        loss.append((y_pred.item() - labels[im].item()) ** 2)

print("RMSE: ", np.sqrt(np.mean(loss)))

plt.figure()
plt.scatter(all_labels, all_outputs, s=3, c="red")
plt.ylabel('Predicted Velocity (m/s)')
plt.xlabel('Velocity Label (m/s)')

plt.savefig(PRPath)
plt.show()

# plt.figure()
# plt.scatter(all_labels, all_error, s=3, c="red")
# plt.xlabel('Velocity (m/s)')
# plt.ylabel('Absolute Error(m/s)')

# plt.savefig(RMSEPath)
# plt.show()

plt.figure()
timet = np.linspace(0, len(all_outputs) / 20, len(all_outputs))
plt.plot(timet, all_outputs, color="C0")
plt.plot(timet, all_labels, color="black")
plt.legend(["Predicted Velocity", "Actual Velocity"])
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')

plt.savefig(Timeseries)
plt.show()