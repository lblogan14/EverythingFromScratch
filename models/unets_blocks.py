'''Various Implementation of deep learning module blocks
Contributer: Bin Liu
Created Date: 2024-03-09
Last Updated: 2024-03-09

This file contains multiple implementations of deep learning module blocks. These blocks are used to build deep learning models.
Just like building lego blocks, these blocks can be used to build different deep learning models.
Consider to use these blocks to build all deep learning models if possible.
'''

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''Double convolution block
    (conv2d -> BN -> activation) * 2'''
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, dropout: bool =False, padding_mode: str ='zeros'):
        '''Constructor for DoubleConv class
        
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        mid_channels: Optional[int]
            Number of middle channels, default is None
        dropout: bool
            Whether to use dropout, default is False
        padding_mode: str
            Padding mode, default is 'zeros'
        '''
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.use_dropout = dropout
        # if dropout is used
        if self.use_dropout:
            self.drop2d = nn.Dropout2d(p=0.2)

        # two conv layers
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode)

        # two batch normalization layers
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # two activation layers
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass for DoubleConv class
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        if self.use_dropout:
            x = self.drop2d(x)

        x = self.bn2(self.conv2(x))
        x = self.act2(x)

        return x
    

class SimpleResidualBlockConv(nn.Module):
    '''Simple residual block with convolution'''
    def __init__(self, in_channels: int, out_channels: int, dropout: bool =False, padding_mode: str ='zeros'):
        '''Constructor for SimpleResidualBlockConv class
        
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        dropout: bool
            Whether to use dropout, default is False
        padding_mode: str
            Padding mode, default is 'zeros'
        '''
        super(SimpleResidualBlockConv, self).__init__()
        self.use_dropout = dropout
        # if dropout is used
        if self.use_dropout:
            self.drop2d = nn.Dropout2d(p=0.2)

        # two conv layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode)

        # two batch normalization layers
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # two activation layers
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        # identity mapping (skip connection)
        self.identity_mapping = nn.Sequential()
        if in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass for SimpleResidualBlockConv class
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        residual = x

        x = self.bn1(self.conv1(x))
        x = self.act1(x)

        if self.use_dropout:
            x = self.drop2d(x)

        x = self.bn2(self.conv2(x))

        # add identity mapping (skip connection)
        x += self.identity_mapping(residual)
        x = self.act2(x)

        return x
    

class Down(nn.Module):
    '''Downsampling block'''
    def __init__(self, in_channels: int, out_channels: int, blockname: str = 'doubleconv'):
        '''Constructor for Down class
        
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        blockname: str
            Name of block, default is 'doubleconv'
        '''
        super(Down, self).__init__()
        
        self.pool = nn.MaxPool2d(2)

        # select the block
        assert blockname in ['doubleconv', 'residual'], "Selected block is not available"
        if blockname == 'doubleconv':
            self.block = DoubleConv(in_channels, out_channels)
        elif blockname == 'residual':
            self.block = SimpleResidualBlockConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass for Down class
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        x = self.pool(x)
        x = self.block(x)
        return x
    

class Up(nn.Module):
    '''Upsampling block'''
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, blockname: str = 'doubleconv'):
        '''Constructor for Up class
        
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        bilinear: bool
            Whether to use bilinear interpolation, default is True
        blockname: str
            Name of block, default is 'doubleconv'
        '''
        super(Up, self).__init__()

        # select the block
        assert blockname in ['doubleconv', 'residual'], "Selected block is not available"

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if blockname == 'doubleconv':
                self.block = DoubleConv(in_channels, out_channels, in_channels // 2)
            elif blockname == 'residual':
                self.block = SimpleResidualBlockConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if blockname == 'doubleconv':
                self.block = DoubleConv(in_channels, out_channels)
            elif blockname == 'residual':
                self.block = SimpleResidualBlockConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        '''Forward pass for Up class
        
        Parameters
        ----------
        x1: torch.Tensor
            Input tensor to the upsampling block
        x2: torch.Tensor
            Concatenated tensor from the downsampled block

        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        x1 = self.up(x1)
        # input size = [N, C, H, W]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # pad x1 to match the same size as x2
        x1 = F.pad(x1,
                   #[padding_left, padding_right,    padding_top, padding_bottom]
                   [diffX//2,      diffX - diffX//2, diffY//2,    diffY - diffY//2],
                   mode='reflect')
        
        # UNet concatenation
        x = torch.cat([x2, x1], dim=1)
        x = self.block(x)

        return x
    

class OutConv(nn.Module):
    '''Output head block'''
    def __init__(self, in_channels: int, out_channels:int):
        '''Constructor for OutConv class
        
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        '''
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass for OutConv

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        x = self.conv(x)

        return x
    

class ResNetBlock2D(nn.Module):
    '''ResNet block for 2D data'''
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int =3, stride: int =1, padding: int =0, padding_mode:str ='zeros', \
                 mid_channel_multiplier: int =1, dropout: bool =False):
        '''Constructor for ResNetBlock2D class

        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        kernel_size: int
            Size of the kernel, default is 3
        stride: int
            Stride of the kernel, default is 1
        padding: int
            Padding of the kernel, default is 0
        padding_mode: str
            Padding mode, default is 'zeros'
        mid_channel_multiplier: int
            Middle channel multiplier, default is 1 
        dropout: bool
            Whether to use dropout, default is False
        '''
        super(ResNetBlock2D, self).__init__()

        # initialize block layers
        layers = []
        # first normalization and activation
        layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.ReLU())
        # first convolution
        layers.append(nn.Conv2d(in_channels,
                                in_channels * mid_channel_multiplier,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode=padding_mode,))
        
        # if dropout is used
        if dropout:
            layers.append(nn.Dropout2d(p=0.2))

        # second normalization and activation
        layers.append(nn.BatchNorm2d(in_channels * mid_channel_multiplier))
        layers.append(nn.ReLU())
        # second convolution
        layers.append(nn.Conv2d(in_channels * mid_channel_multiplier,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                padding_mode=padding_mode))
        
        # create the block
        self.block = nn.Sequential(*layers)
        # identity mapping (skip connection)
        self.identity_mapping = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.identity_mapping = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass for ResNetBlock2D class

        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        
        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        out = self.block(x)
        # add identity mapping (skip connection)
        out += self.identity_mapping(out)
        return out