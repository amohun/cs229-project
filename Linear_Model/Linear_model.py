import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision
import pickle as pkl
import itertools

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