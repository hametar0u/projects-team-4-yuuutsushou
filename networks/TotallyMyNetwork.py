import torch
import torch.nn as nn


class TotallyMyNetwork(torch.nn.Module):
    def __init__(self, dropout_pct=0, model_name='resnet18'):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=7, stride=2, padding=3)
        tempmodel = torch.hub.load('pytorch/vision:v0.9.0', model_name, pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(tempmodel.children())[:-1]))
        
        self.fc1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_pct)
        self.bnfc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.resnet(x)
        x = torch.squeeze(x)  # flatten
        x = self.relu(self.fc1(x))
        x = self.bnfc1(self.dropout(x))
        x = self.fc2(x)
        return x