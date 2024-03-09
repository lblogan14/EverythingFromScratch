'''Various Implementation of U-Net and its variants
Contributer: Bin Liu
Created Date: 2024-03-09
Last Updated: 2024-03-09

This file may contain multiple implementations of U-Net and its variants but with different methods of implementation.
All U-Net class implementations should take in an encoder instance and an decoder instance as input and return a U-Net model instance.
'''

from typing import Dict, Tuple
import torch
import torch.nn as nn



class VanillaUNet(nn.Module):
    '''Vanilla U-Net'''
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        '''Constructor for VanillaUNet class
        
        Parameters
        ----------
        encoder: nn.Module
            Encoder instance
        decoder: nn.Module
            Decoder instance
        '''
        super(VanillaUNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> \
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        '''Forward pass for Vanilla UNet class
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor
        
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            Tuple of output tensor, encoder features and decoder features
        '''
        # encoding path
        x, enc_features = self.encoder(x)
        # decoding path
        x, dec_features = self.decoder(x, enc_features)

        return x, enc_features, dec_features