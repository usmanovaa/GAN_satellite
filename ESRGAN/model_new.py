import math
import torch
from torch import nn

from torchvision.models import vgg19


import math
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, scale_factor, n_basic_block =3):

        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU())
        
        basic_block_layer = []
        for _ in range(n_basic_block):
            basic_block_layer += [RRDB(64, 32)]

        self.basic_block = nn.Sequential(*basic_block_layer)
        
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.upsample = upsample_block(64, scale_factor)
        
        self.conv3 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=9, padding=4))

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)
        x = self.upsample(x + x1)
        x = self.conv3(x)
        return (torch.tanh(x) + 1) / 2 
    
    
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf = 64, gc = 32, bias = True):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding = 1, bias = bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding = 1, bias = bias)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, padding = 1, bias = bias)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, padding = 1, bias = bias)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, padding = 1, bias = bias)
        self.lrelu = nn.LeakyReLU(negative_slope = 0.2, inplace = True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1),1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2),1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3),1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4),1))

        return x5*0.2 + x
    
    
class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''
    def __init__(self, nf, gc=32):
        
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)
    
    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x
    
    
def upsample_block(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor//2):
        block += [
            nn.Conv2d(nf, nf * (2 ** 2), 1),
            nn.PixelShuffle(2),
            nn.ReLU()
        ]

    return nn.Sequential(*block)

    #####################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1), #???
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size)), self.net(x)
        #return torch.sigmoid(self.net(x).view(batch_size))