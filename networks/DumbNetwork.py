import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import data.StartingDataset

# the whole point of pytorch is to be able to easily train models on GPU's and TPU's
# if you have a dedicated GPU, you can download CUDA from Nvidia toolkits
# this snippet of code will check if you have cuda installed, and will set the 'device' accordingly
# for now, it should probably return "Running on CPU"

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on GPU')
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print('Running on CPU')

# TODO: Do not run this this is different data

# train_dataset = torchvision.datasets.CIFAR10(root='./cifar10', transform=torchvision.transforms.ToTensor(), download=True)
# test_dataset = torchvision.datasets.CIFAR10(root='./cifar10', transform=torchvision.transforms.ToTensor(), download=True, train=False)

# create training and testing loaders

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

data_loader = data.StartingDataset.StartingDataset()

# train_iter = iter(train_loader) # convert your train loader to an iterator

# batch_images, batch_labels = next(train_iter)
# image, label = batch_images[0], batch_labels[0]

# construct your CNN model

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # input dim: 3x32x32
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        # output dim: 6x32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # output dim: 6x16x16
        self.c2 = nn.Conv2d(6, 16, 5, 1, 2)
        # output dim: 16x16x16
        self.pool2 = nn.MaxPool2d(2, 2)
        # output dim: 16x8x8
        self.flat = nn.Flatten()
        self.d1 = nn.Linear(480000, 120)  # Behold, the CNN - Confused Neural Network
        self.d2 = nn.Linear(120, 84)
        self.d3 = nn.Linear(84, 5)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.c1(x)))
        x = self.pool2(F.relu(self.c2(x)))
        x = self.flat(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x) 
        x = self.softmax(x)
        # print(x)
        # print("Size: ", x.size())
        return x