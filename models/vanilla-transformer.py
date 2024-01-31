'''The original Transformer model from the paper "Attention is All You Need" (Vaswani et al., 2017)
Contributer: Bin Liu
Created Date: 2024-01-30
Last Updated: 2024-01-30

This file may contain multiple implementations of the original Transformer model but with different methods of implementation.
The goal is to provide different ways to implement the same model to help readers understand the model better.
'''

# Typical imports
import torch
from torch import nn
import torch.nn.functional as F
import math

######################################### Multi-head Attention Class #########################################
'''The multi-head attention mechanism is the core of the Transformer model. 
It is used to capture the relationship between different words in the input sequence.
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        '''Initialize the multi-head attention mechanism.
        
        Parameters:
        ----------
        d_model: int
            The dimension of the input sequence.
        num_heads: int
            The number of heads to split the input sequence.
        '''
        super(MultiHeadAttention, self).__init__()
        # Check if the input dimension is divisible by the number of heads
        assert d_model % num_heads == 0, 'The input dimension (`d_model`) must be divisible by the number of heads (`num_head`).'
        # seq_len = sequence length = number of words in the input sequence
        # d_model = dimension of the input sequence = dimension of the word embeddings

        # Initialize the parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Initialize the linear layers for the query, key, value, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)


    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        '''The scaled dot-product attention mechanism.
        
        Parameters:
        ----------
        Q: torch.Tensor
            The query tensor with shape (batch_size, num_heads, seq_len, d_k).
        K: torch.Tensor
            The key tensor with shape (batch_size, num_heads, seq_len, d_k).
        V: torch.Tensor
            The value tensor with shape (batch_size, num_heads, seq_len, d_k).
        mask: torch.Tensor
            The mask tensor with shape (batch_size, seq_len, seq_len).
            
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, num_heads, seq_len, d_k).
        '''
        # Compute the attention scores: Q * K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5) # (batch_size, num_heads, seq_len, seq_len)

        # Check if the mask is provided
        if mask is not None:
            # Apply the mask to the attention scores
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Compute the attention weights
        attn_probs = torch.softmax(attn_scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        # Compute the output tensor
        output = torch.matmul(attn_probs, V) # (batch_size, num_heads, seq_len, d_k)

        return output
    
    
    def split_heads(self, x):
        '''Split the input tensor into multiple heads.
        
        Parameters:
        ----------
        x: torch.Tensor
            The input tensor with shape (batch_size, seq_len, d_model).
        
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, num_heads, seq_len, d_k).
        '''
        # Get the size of the input tensor
        batch_size, seq_length, d_model = x.size()
        # Reshape the input tensor to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        # Transpose the tensor to (batch_size, num_heads, seq_len, d_k)
        x = x.transpose(1, 2)

        return x
    
    def combine_heads(self, x):
        '''Combine the heads into a single tensor.
        
        Parameters:
        ----------
        x: torch.Tensor
            The input tensor with shape (batch_size, num_heads, seq_len, d_k).
            
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, seq_len, d_model).
        '''
        # Get the size of the input tensor
        batch_size, num_heads, seq_length, d_k = x.size()
        # Transpose the tensor to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        # Reshape the tensor to (batch_size, seq_len, d_model)
        x = x.contiguous().view(batch_size, seq_length, self.d_model)
        '''The `contiguous()` method is used to ensure that the tensor is stored in a contiguous block of memory before applying the `view()` method.
        The `view()` method is used to reshape the tensor, but it requires the underlying data to be stored in a contiguous block of memory. 
        A contiguous tensor means that the elements are stored in a sequential order in memory without any gaps or irregularities.
        If the tensor is not stored contiguously, calling the `view()` method will raise an error. 
        In such cases, we need to make the tensor contiguous using the `contiguous()` method before applying the `view()` method.
        By calling `x.contiguous()`, we ensure that the tensor x is stored in a contiguous block of memory, allowing us to reshape it using the view() method without any issues.'''

        return x
    

    def forward(self, Q, K, V, mask=None):
        '''The forward pass of the multi-head attention mechanism.
        
        Parameters:
        ----------
        Q: torch.Tensor
            The query tensor with shape (batch_size, seq_len, d_model).
        K: torch.Tensor
            The key tensor with shape (batch_size, seq_len, d_model).
        V: torch.Tensor
            The value tensor with shape (batch_size, seq_len, d_model).
        mask: torch.Tensor
            The mask tensor with shape (batch_size, seq_len, seq_len).
            
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, seq_len, d_model).
        '''
        # Apply the linear layers to the query, key, and value tensors
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        # Split the query, key, and value tensors into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply the scaled dot-product attention mechanism
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # Combine the heads into a single tensor
        attn_output = self.combine_heads(attn_output)
        # Apply the output linear layer
        output = self.W_o(attn_output)

        return output
    

######################################### Position-wise Feedforward Network Class #########################################
'''The position-wise feedforward network is used to capture the local information within the input sequence.
It is applied to each position in the input sequence independently and identically.
'''
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        '''Initialize the position-wise feedforward network.
        
        Parameters:
        ----------
        d_model: int
            The dimension of the input sequence.
        d_ff: int
            The dimension of the feedforward network.
        '''
        super(PositionWiseFeedForward, self).__init__()
        # Initialize the linear layers
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # Initialize the activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        '''The forward pass of the position-wise feedforward network.
        
        Parameters:
        ----------
        x: torch.Tensor
            The input tensor with shape (batch_size, seq_len, d_model).
            
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, seq_len, d_model).
        '''
        x = self.fc1(x)
        x = self.activation(x)
        o = self.fc2(x)

        return o
    

######################################### Positional Encoding Class #########################################
'''The positional encoding is used to inject the position information into the input sequence.
It is added to the input sequence before feeding it into the multi-head attention mechanism.
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        '''Initialize the positional encoding.
        
        Parameters:
        ----------
        d_model: int
            The dimension of the input sequence.
        max_seq_length: int
            The maximum length of the input sequence.
        '''
        super(PositionalEncoding, self).__init__()

        # Create the positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        # Create a vector of the same size as the positional encoding matrix
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # (max_seq_length, 1)
        # Compute the positional encoding values
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)
        # Register the positional encoding matrix as a buffer
        self.register_buffer('pe', pe)
        '''The `register_buffer` method provides a way to add tensors as buffers to a PyTorch module, 
        allowing us to store and manipulate additional information that is not considered a model parameter.'''

    def forward(self, x):
        '''The forward pass of the positional encoding.
        
        Parameters:
        ----------
        x: torch.Tensor
            The input tensor with shape (batch_size, seq_len, d_model).
        
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, seq_len, d_model).
        '''
        # Add the positional encoding to the input tensor
        x = x + self.pe[:, :x.size(1)]
        '''The positional encoding matrix is added to the input tensor x.
        The positional encoding matrix is sliced to match the length of the input sequence,
        and then added to the input tensor x.'''