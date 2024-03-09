'''Various Implementation of encoder for U-Net and its variants
Contributer: Bin Liu
Created Date: 2024-03-09
Last Updated: 2024-03-09

This file may contain multiple implementations of encoder classes for U-Net and its variants but with different methods of implementation.
All encoder class implementations should consider to use the blocks from EverythingFromScratch/models/unet_blocks.py if possible.
'''
from typing import Dict, Tuple
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .unets_blocks import *


class VanillaUNetEncoder(nn.Module):
    '''Vanilla U-Net encoder'''
    def __init__(self, in_channels: int, num_blocks: int, base_channels: int, channel_multiplier: int = 2, bilinear: str = False):
        '''Constructor for VanillaUNetEncoder class
        
        Parameters
        ----------
        in_channels: int
            Number of input channels
        num_blocks: int
            Number of blocks
        base_channels: int
            Number of base channels
        channel_multiplier: int
            Channel multiplier, default is 2
        bilinear: str
            Whether to use bilinear, default is False
        '''
        super(VanillaUNetEncoder, self).__init__()
        
        # create parameters
        channel_list = [base_channels * channel_multiplier ** i for i in range(num_blocks)]
        factor = 2 if bilinear else 1

        # create blocks
        self.blocks = nn.ModuleDict()

        # input block (block_0)
        self.blocks['block_0'] = DoubleConv(in_channels, base_channels)
        # encoding path
        for i in range(num_blocks):
            if i != num_blocks - 1:
                # not the last block
                self.blocks[f'encoder_block_{i+1}'] = Down(channel_list[i],
                                                           channel_list[i+1])
            else:
                # the last block
                self.blocks[f'encoder_block_{i+1}'] = Down(channel_list[i],
                                                           channel_list[i] * channel_multiplier // factor)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''Forward pass for VanillaUNetEncoder class
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Output tensor and feature map tensors
        '''
        enc_features = {}
        for name, block in self.blocks.items():
            x = block(x)
            enc_features[name] = x

        return x, enc_features
    

class ResUNet2DEncoder(nn.Module):
    '''Residual U-Net 2D encoder'''
    def __init__(self, in_channels: int, num_blocks: int, base_channels: int = 32, channel_multiplier:int = 1,
                 in_ksize: int = 3, in_stride: int = 1, in_padding: int = 1,
                 block_ksize: int = 3, block_stride: int = 1, block_padding: int = 1,
                 block_mid_channel_multiplier: int = 1, block_ksize_increment: int = 0,
                 padding_mode='zeros', dropout: bool = False):
        '''Constructor for ResUNet2DEncoder class

        Parameters
        ----------
        in_channels: int
            Number of input channels
        num_blocks: int
            Number of blocks
        base_channels: int
            Number of base channels, default is 32
        channel_multiplier: int
            Channel multiplier for each block, default is 1
        in_ksize: int
            Kernel size for the input convolution, default is 3
        in_stride: int
            Stride for the input convolution, default is 1
        in_padding: int
            Padding for the input convolution, default is 1
        block_ksize: int
            Kernel size for the residual blocks, default is 3
        block_stride: int
            Stride for the residual blocks, default is 1
        block_padding: int
            Padding for the residual blocks, default is 1
        block_mid_channel_multiplier: int
            Channel multiplier for the middle convolution in the residual blocks, default is 1
        block_ksize_increment: int
            Increment for the kernel size in the residual blocks, default is 0
        padding_mode: str
            Padding mode for the convolutions, default is 'zeros'
        dropout: bool
            Whether to use dropout, default is False
        '''
        super(ResUNet2DEncoder, self).__init__()

        # `block_ksize_increment` should be even
        assert block_ksize_increment % 2 == 0, 'block_ksize_increment should be even'

        # create parameters
        channel_list = [base_channels * channel_multiplier ** i for i in range(num_blocks+1)]
        # len(channel_list) == num_blocks + 1
        # so that the last element is the number of channels for the output of the encoder

        # create block parameters
        block_ksize_list = [block_ksize + block_ksize_increment * i for i in range(num_blocks)]
        block_stride_list = [block_stride] * num_blocks
        block_padding_list = []
        for i in range(num_blocks):
            if block_ksize_increment == 0:
                block_padding_list.append(block_padding)
            else:
                block_padding_list.append(block_padding + block_ksize_increment * i // 2)

        # create blocks
        self.blocks = nn.ModuleDict()

        # input block
        self.blocks['encoder_block_0'] = nn.Conv2d(in_channels,
                                                   base_channels,
                                                   in_ksize,
                                                   in_stride,
                                                   in_padding,
                                                   padding_mode=padding_mode)
        
        # encoding path
        for i in range(num_blocks):
            self.blocks[f'encoder_block_{i+1}'] = ResNetBlock2D(
                in_channels=channel_list[i],
                out_channels=channel_list[i+1],
                kernel_size=block_ksize_list[i],
                stride=block_stride_list[i],
                padding=block_padding_list[i],
                padding_mode=padding_mode,
                mid_channel_multiplier=block_mid_channel_multiplier,
                dropout=dropout,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''Forward pass for ResUNet
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Output tensor and feature map tensors, 
            where the keys are the names of the blocks and the values are the output tensors
        '''
        enc_features = {}
        for name, block in self.blocks.items():
            x = block(x)
            enc_features[name] = x

        return x, enc_features