import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=100):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.num_classes = num_classes

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        fc = self.classificate(out)
        out = fc(out)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
    
    def classificate(self, out):
        classificate = nn.Sequential(
            nn.Linear(out.size(1), 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.num_classes),
        )
        return classificate
    




def test():
    for i in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
        net = VGG(f'{i}')
        x = torch.randn(2, 3, 224, 224)
        y = net(x)
        print(y)

test()
