import torch
import torch.nn as nn
import torch.nn.functional as F

#using_Data is MNIST

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5) 
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.tanh(self.c1(x))
        x = F.avg_pool2d(x, 2)
        x = F.tanh(self.c3(x))
        x = F.avg_pool2d(x, 2)
        x = F.tanh(self.c5(x))
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

