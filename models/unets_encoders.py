'''Various Implementation of encoder for U-Net and its variants
Contributer: Bin Liu
Created Date: 2024-03-09
Last Updated: 2024-03-09

This file may contain multiple implementations of encoder classes for U-Net and its variants but with different methods of implementation.
All encoder class implementations should consider to use the blocks from EverythingFromScratch/models/unet_blocks.py if possible.
'''
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