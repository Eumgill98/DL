import torch.nn as nn
import torch.nn.functional as F

#using STL-10 dataset

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            #layer 1
            nn.Conv2d(3, 96, kernel_size=11, stride = 4, padding = 0),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k = 2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            #layer 2
            nn.Conv2d(96, 256, kernel_size=5, stride = 1, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            #layer 3
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),

            #layer 4
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),

            #layer 5
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace = False),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace= False),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )   

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x
