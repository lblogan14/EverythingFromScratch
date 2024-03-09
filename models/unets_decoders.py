'''Various Implementation of decoder for U-Net and its variants
Contributer: Bin Liu
Created Date: 2024-03-09
Last Updated: 2024-03-09

This file may contain multiple implementations of decoder classes for U-Net and its variants but with different methods of implementation.
All decoder class implementations should consider to use the blocks from EverythingFromScratch/models/unet_blocks.py if possible.
'''

from typing import Dict, Tuple
import torch
import torch.nn as nn
from .unets_blocks import *

class VanillaUNetDecoder(nn.Module):
    '''Vanilla U-Net decoder'''
    def __init__(self, num_classes: int, num_blocks: int, base_channels: int, channel_multiplier: int = 2, bilinear: str = False):
        '''Constructor for VanillaUNetDecoder class
        
        Parameters
        ----------
        num_classes: int
            Number of classes
        num_blocks: int
            Number of blocks
        base_channels: int
            Number of base channels
        channel_multiplier: int
            Channel multiplier, default is 2
        bilinear: str
            Whether to use bilinear, default is False
        '''
        super(VanillaUNetDecoder, self).__init__()
        
        # create parameters
        channel_list = [base_channels * channel_multiplier ** i for i in range(num_blocks)]
        factor = 2 if bilinear else 1

        # create blocks
        self.blocks = nn.ModuleDict()

        # decoding path
        # iterate through the blocks in reverse order
        for i in range(num_blocks - 1, -1, -1):
            if i != 0:
                # not the last blocks
                self.blocks[f'decoder_block_{i}'] = Up(channel_list[i] * channel_multiplier,
                                                       channel_list[i] // factor,
                                                       bilinear)
            else:
                # the last block
                self.blocks[f'decoder_block_{i}'] = Up(channel_list[i] * channel_multiplier,
                                                       channel_list[i],
                                                       bilinear)
                
        # output block
        self.blocks['decoder_output'] = OutConv(channel_list[0], num_classes)

    def forward(self, x: torch.Tensor, enc_features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''Forward pass for VanillaUNetDecoder class
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor from the encoder output
        enc_features: Dict[str, torch.Tensor]
            Dictionary of encoder features; keys are the names of the blocks and values are the output tensors
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            Output tensor and feature map tensors
        '''
        dec_features = {}
        for name, block in self.blocks.items():
            if 'decoder_output' in name:
                # output block
                x = block(x)
            else:
                # concatenate the feature map from the encoder
                # there is an index shift between the encoder and decoder blocks
                idx = int(name.split('_')[-1])
                x = block(x, enc_features[f'encoder_block_{idx-1}'])
            dec_features[name] = x

        return x, dec_features