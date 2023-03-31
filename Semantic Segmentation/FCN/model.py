import torch 
from torch import nn
from torchvision.models.vgg import VGG
from torchvision import models

#backbone use VGG 

class FCN(nn.Module):
    def __init__(self, backbone, model_type='FCN32', num_class=2):
        super().__init__()
        self.model_type = model_type

        self.backbone = backbone
        
        self.relu = nn.ReLU(inplace=True)
        
        

        self.conv_out1 = nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1)
        self.conv_out2 = nn.Conv2d(512, 512, kernel_size=1, padding=0, stride=1)
        self.conv_out3 = nn.Conv2d(512, num_class, kernel_size=1, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(num_class)
        self.upsample32 = torch.nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.upsample16 = torch.nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False )
        self.upsample8 =  torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample2_1 = torch.nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False )
        self.upsample2_2 = torch.nn.Upsample(scale_factor=2,mode='bilinear', align_corners=False )

        self.backbone_one1 = nn.Conv2d(512, num_class, kernel_size=1, padding=0, stride=1)
        self.backbone_one2 = nn.Conv2d(256, num_class, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        backbone_out = self.backbone(x)
        x3 = backbone_out['x3']
        x4 = backbone_out['x4']
        x5 = backbone_out['x5']
        
        if self.model_type=='FCN32':
            x = self.bn1(self.relu(self.conv_out1(x5)))
            x = self.bn2(self.relu(self.conv_out2(x)))
            x = self.bn3(self.relu(self.conv_out3(x)))
            x = self.upsample32(x)
        
        elif self.model_type=='FCN16':
            x_ = self.backbone_one1(x4)

            x = self.bn1(self.relu(self.conv_out1(x5)))
            x = self.bn2(self.relu(self.conv_out2(x)))
            x = self.bn3(self.relu(self.conv_out3(x)))
            x = self.upsample2_1(x)
            
            x = x + x_
            x = self.upsample16(x)

        elif self.model_type=='FCN8':
            x_ = self.backbone_one1(x4)
            x__ = self.backbone_one2(x3)

            x = self.bn1(self.relu(self.conv_out1(x5)))
            x = self.bn2(self.relu(self.conv_out2(x)))
            x = self.bn3(self.relu(self.conv_out3(x)))
            x = self.upsample2_1(x)
            x = x + x_
            x = self.upsample2_2(x)
            x = x__ + x
            x = self.upsample8(x)
        
        return x




class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)





