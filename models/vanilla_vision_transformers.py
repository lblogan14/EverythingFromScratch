'''The original Vision Transformer model from the paper 
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" 
by Alexey Dosovitskiy et al. (2021)

Contributer: Bin Liu
Created Date: 2024-02-05
Last Updated: 2024-02-05

This file may contain multiple implementations of the original Vision Transformer model but with different methods of implementation.
The goal is to provide different ways to implement the same model to help readers understand the model better.
'''

# Typical imports
import torch
import torch.nn as nn
import torch.nn.functional as F

