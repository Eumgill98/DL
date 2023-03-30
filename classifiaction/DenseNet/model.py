import torch
from torch import nn
from torchsummary import summary

#model config : [[layer num], grow_rate]
__model__ = {
    'Densenet121' : [[6,12,24,16], 32],
    'Densenet169' : [[6,12,32,32], 32],
    'Densenet161' : [[6,12,36,24], 48],
    'Densenet201' : [[6,12,48,32], 32]
}
    

#model define
class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleNeck, self).__init__()
        inter_channel = 4 * growth_rate

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottleneck(x)], 1)
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)
    

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        inter_channels = 2 * growth_rate
        
        self.conv1 = nn.Conv2d(3, inter_channels, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.mini_model = nn.Sequential()

        for idx in range(len(nblocks) - 1):
            self.mini_model.add_module(f"dense_block_{idx}", self._make_layers(block, inter_channels, nblocks[idx]))
            inter_channels += growth_rate * nblocks[idx]
            out_channels = int(reduction * inter_channels)
            self.mini_model.add_module(f"transition_layer{idx}", Transition(inter_channels, out_channels))
            inter_channels = out_channels

        self.mini_model.add_module(f"dense_block{len(nblocks) - 1}", self._make_layers(block, inter_channels, nblocks[-1]))
        inter_channels += growth_rate * nblocks[-1]
        self.mini_model.add_module('bn', nn.BatchNorm2d(inter_channels))
        self.mini_model.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(inter_channels, num_class)            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.mini_model(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


    def _make_layers(self, block, in_chaanels, nblocks):
        dense_block = nn.Sequential()
        for idx in range(nblocks):
            dense_block.add_module(f'botte_neck{idx}', block(in_chaanels, self.growth_rate))
            in_chaanels += self.growth_rate
        return dense_block
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    




#make model
'''
model:str -> (Densenet121, Densenet161, Densenet169, Densenet201)
'''
def make_model(model:str, num_class = 100):
    return DenseNet(BottleNeck, __model__[model][0], __model__[model][1], num_class=num_class)


#test
# x = torch.randn(3, 3, 32, 32)
# model = make_model('Densenet121', num_class=10)  
# output = model(x)
# print(output.size())

#summury
#model = make_model('Densenet121')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# summary(model.to(device), (3,224,224), device=device.type)