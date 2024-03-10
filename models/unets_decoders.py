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
    

class ResUNet2DDecoder(nn.Module):
    '''Residual U-Net 2D decoder'''
    def __init__(self, out_channels: int, num_blocks: int, base_channels: int = 32, channel_multiplier: int = 1,
                 out_ksize: int = 3, out_stride: int = 1, out_padding: int = 1,
                 block_ksize: int = 3, block_stride: int = 1, block_padding: int = 1,
                 block_mid_channel_multiplier: int = 1, block_ksize_increment: int = 0,
                 padding_mode: str = 'zeros', dropout: bool = False):
        '''Constructor for ResUNet2DDecoder class
        
        Parameters
        ----------
        out_channels: int
            Number of output channels
        num_blocks: int
            Number of blocks
        base_channels: int
            Number of base channels, default is 32
        channel_multiplier: int
            Channel multiplier, default is 1
        out_ksize: int
            Kernel size for the output block, default is 3
        out_stride: int
            Stride for the output block, default is 1
        out_padding: int
            Padding for the output block, default is 1
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
        super(ResUNet2DDecoder, self).__init__()

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
        # concatenation layers after adding encoder features
        self.concat = nn.ModuleDict()

        # decoding path
        # two sequential tasks:
        # 1. passing through a decoder block
        # 2. concatenating with the corresponding encoder feature map
        for i in range(num_blocks-1, -1, -1):
            # traverse through the decoder blocks in reverse order

            # create decoder block
            self.blocks[f'decoder_block_{i+1}'] = ResNetBlock2D(
                in_channels=channel_list[i+1],
                out_channels=channel_list[i],
                kernel_size=block_ksize_list[i],
                stride=block_stride_list[i],
                padding=block_padding_list[i],
                padding_mode=padding_mode,
                mid_channel_multiplier=block_mid_channel_multiplier,
                dropout=dropout
            )

            # create concatenation layer
            # NOTE: in_channels = 2 * out_channels of the decoder block
            self.concat[f'decoder_concat_{i+1}'] = nn.Conv2d(
                in_channels=channel_list[i] * 2,
                out_channels=channel_list[i],
                kernel_size=1,
                stride=1,
                padding=0,
            )

        # output block
        self.blocks['decoder_output'] = nn.Conv2d(
            in_channels=channel_list[0],
            out_channels=out_channels,
            kernel_size=out_ksize,
            stride=out_stride,
            padding=out_padding,
            padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor, enc_features: Dict[str, torch.Tensor]) \
        -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        '''Forward pass for ResUNet2DDecoder class
        
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
                # going through a decoder block
                x = block(x)
                # concatenating with the corresponding encoder feature map
                # there is an index shift between the encoder and decoder blocks
                idx = int(name.split('_')[-1])
                x = torch.cat([x, enc_features[f'encoder_block_{idx-1}']], dim=1)
                x = self.concat[f'decoder_concat_{idx}'](x)

            dec_features[name] = x

        return x, dec_features
    

class ResUNet2DDecoder_Attention(nn.Module):
    '''Residual U-Net 2D decoder with attention'''
    def __init__(self, out_channels: int, num_blocks: int, base_channels: int = 32, channel_multiplier: int = 1,
                 out_ksize: int = 3, out_stride: int = 1, out_padding: int = 1,
                 block_ksize: int = 3, block_stride: int = 1, block_padding: int = 1,
                 block_mid_channel_multiplier: int = 1, block_ksize_increment: int = 0,
                 padding_mode: str = 'zeros', dropout: bool = False,
                 attn_channels: int = 8):
        '''Constructor for ResUNet2DDecoder_Attention class
        
        Parameters
        ----------
        out_channels: int
            Number of output channels
        num_blocks: int
            Number of blocks
        base_channels: int
            Number of base channels, default is 32
        channel_multiplier: int
            Channel multiplier, default is 1
        out_ksize: int
            Kernel size for the output block, default is 3
        out_stride: int
            Stride for the output block, default is 1
        out_padding: int
            Padding for the output block, default is 1
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
        attn_channels: int
            Number of channels for the attention mechanism, default is 8
        '''
        super(ResUNet2DDecoder_Attention, self).__init__()

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
        # attention layers after adding encoder features
        self.attn = nn.ModuleDict()
        # concatenation layers after adding attention
        self.concat = nn.ModuleDict()

        # decoding path
        # three sequential tasks:
        # 1. passing through a decoder block
        # 2. adding attention with the corresponding encoder feature map
        # 3. concatenating with the corresponding encoder feature map after attention
        for i in range(num_blocks-1, -1, -1):
            # traverse through the decoder blocks in reverse order

            # create decoder block
            self.blocks[f'decoder_block_{i+1}'] = ResNetBlock2D(
                in_channels=channel_list[i+1],
                out_channels=channel_list[i],
                kernel_size=block_ksize_list[i],
                stride=block_stride_list[i],
                padding=block_padding_list[i],
                padding_mode=padding_mode,
                mid_channel_multiplier=block_mid_channel_multiplier,
                dropout=dropout
            )

            # create attention layer
            # assuming the encoder and decoder have the same number of channels
            self.attn[f'decoder_attn_{i+1}'] = AttentionGate2D(
                F_g=channel_list[i],
                F_l=channel_list[i],
                F_int=attn_channels
            )

            # create concatenation layer
            # NOTE: in_channels = 2 * out_channels of the decoder block
            self.concat[f'decoder_concat_{i+1}'] = nn.Conv2d(
                in_channels=channel_list[i] * 2,
                out_channels=channel_list[i],
                kernel_size=1,
                stride=1,
                padding=0,
            )

        # output block
        self.blocks['decoder_output'] = nn.Conv2d(
            in_channels=channel_list[0],
            out_channels=out_channels,
            kernel_size=out_ksize,
            stride=out_stride,
            padding=out_padding,
            padding_mode=padding_mode
        )

    def forward(self, x: torch.Tensor, enc_features: Dict[str, torch.Tensor]) \
        -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        '''Forward pass for ResUNet2DDecoderWithAttention class
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor from the encoder output
        enc_features: Dict[str, torch.Tensor]
            Dictionary of encoder features; keys are the names of the blocks and values are the output tensors
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            Output tensor, decoder feature map tensors and attention tensors
        '''
        dec_features = {}
        attn_features = {}
        for name, block in self.blocks.items():
            if 'decoder_output' in name:
                # output block
                x = block(x)
            else:
                # going through a decoder block
                x = block(x)
                # adding attention with the corresponding encoder feature map
                # there is an index shift between the encoder and decoder blocks
                idx = int(name.split('_')[-1])
                x_attn, attn = self.attn[f'decoder_attn_{idx}'](x, enc_features[f'encoder_block_{idx-1}'])

                # Update attention features
                attn_features[f'decoder_attn_{idx}'] = attn

                # concatenating with the corresponding encoder feature map after attention
                x = torch.cat([x, x_attn], dim=1)
                x = self.concat[f'decoder_concat_{idx}'](x)

            # Update decoder features
            dec_features[name] = x

        return x, dec_features, attn_features
