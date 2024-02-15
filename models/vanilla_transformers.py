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

        return x
    

######################################### Transformer Encoder Layer Class #########################################
'''The Transformer encoder layer is used to capture the relationship between different words in the input sequence.
It consists of a multi-head attention mechanism, a position-wise feedforward network, and residual connections with layer normalization.
'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''Initialize the Transformer encoder layer.
        
        Parameters:
        ----------
        d_model: int
            The dimension of the input sequence.
        num_heads: int
            The number of heads to split the input sequence.
        d_ff: int
            The dimension of the feedforward network.
        dropout: float
            The dropout rate.
        '''
        super(TransformerEncoderLayer, self).__init__()
        # Initialize the multi-head attention mechanism (self-attention mechanism)
        self.self_attn = MultiHeadAttention(d_model, num_heads) # custom class defined above
        # Initialize the position-wise feedforward network
        self.feedforward = PositionWiseFeedForward(d_model, d_ff) # custom class defined above
        # Initialize the layer normalizations
        self.norm1 = nn.LayerNorm(d_model) # after the multi-head attention mechanism
        self.norm2 = nn.LayerNorm(d_model) # after the position-wise feedforward network
        # Initialize the dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        '''The forward pass of the Transformer encoder layer.
        
        Parameters:
        ----------
        x: torch.Tensor
            The input tensor with shape (batch_size, seq_len, d_model).
        mask: torch.Tensor
            The mask tensor with shape (batch_size, seq_len, seq_len).
            
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, seq_len, d_model).
        '''
        # Apply the mult-head self-attention mechanism
        attn_output = self.self_attn(x, # query
                                     x, # key
                                     x, # value
                                     mask, # mask
                                     )
        # Apply dropout and add the residual connection
        x = self.dropout(attn_output) + x
        # Apply layer normalization
        x = self.norm1(x)
        # Apply the position-wise feedforward network
        ff_output = self.feedforward(x)
        # Apply dropout and add the residual connection
        x = self.dropout(ff_output) + x
        # Apply layer normalization
        x = self.norm2(x)

        return x
    

######################################### Transformer Decoder Layer Class #########################################
'''The Transformer decoder layer is used to capture the relationship between different words in the input sequence and the output sequence.
It consists of two multi-head attention mechanisms, a position-wise feedforward network, and residual connections with layer normalization.
'''
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''Initialize the Transformer decoder layer.
        
        Parameters:
        ----------
        d_model: int
            The dimension of the input sequence.
        num_heads: int
            The number of heads to split the input sequence.
        d_ff: int
            The dimension of the feedforward network.
        dropout: float
            The dropout rate.
        '''
        super(TransformerDecoderLayer, self).__init__()
        # Initialize the masked multi-head attention mechanism (self-attention mechanism)
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads) # custom class defined above
        # Initialize the multi-head attention mechanism (encoder-decoder cross-attention mechanism)
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # custom class defined above
        # Initialize the position-wise feedforward network
        self.feedforward = PositionWiseFeedForward(d_model, d_ff) # custom class defined above
        # Initialize the layer normalizations
        self.norm1 = nn.LayerNorm(d_model) # after the masked multi-head attention mechanism
        self.norm2 = nn.LayerNorm(d_model) # after the encoder-decoder cross-attention mechanism
        self.norm3 = nn.LayerNorm(d_model) # after the position-wise feedforward network
        # Initialize the dropout layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, source_mask, target_mask):
        '''The forward pass of the Transformer decoder layer.
        
        Parameters:
        ----------
        x: torch.Tensor
            The input tensor with shape (batch_size, seq_len, d_model).
        encoder_output: torch.Tensor
            The output tensor from the Transformer encoder layer with shape (batch_size, seq_len, d_model).
        source_mask: torch.Tensor
            The mask tensor for the input sequence with shape (batch_size, seq_len, seq_len).
        target_mask: torch.Tensor
            The mask tensor for the output sequence with shape (batch_size, seq_len, seq_len).
        
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, seq_len, d_model).
        '''
        # Apply the masked multi-head self-attention mechanism with the target mask
        attn_output = self.masked_self_attn(x, # query
                                            x, # key
                                            x, # value
                                            target_mask, # mask
                                            )
        # Apply dropout and add the residual connection
        x = self.dropout(attn_output) + x
        # Apply layer normalization
        x = self.norm1(x)
        # Apply the cross-attention mechanism with the encoder output and the source mask
        cross_attn_output = self.cross_attn(x, # query,
                                            encoder_output, # key
                                            encoder_output, # value
                                            source_mask, # mask
                                            )
        # Apply dropout and add the residual connection
        x = self.dropout(cross_attn_output) + x
        # Apply layer normalization
        x = self.norm2(x)
        # Apply the position-wise feedforward network
        ff_output = self.feedforward(x)
        # Apply dropout and add the residual connection
        x = self.dropout(ff_output) + x
        # Apply layer normalization
        x = self.norm3(x)

        return x
    

######################################### Transformer Class #########################################
'''The Transformer model is used to capture the relationship between different words in the input sequence and the output sequence.
It consists of a stack of Transformer encoder layers and Transformer decoder layers.
'''
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        '''Initialize the Transformer model.
        
        Parameters:
        ----------
        source_vocab_size: int
            The size of the source vocabulary.
        target_vocab_size: int
            The size of the target vocabulary.
        d_model: int
            The dimension of the input sequence, the word embeddings.
        num_heads: int
            The number of heads to split the input sequence.
        num_layers: int
            The number of layers in the Transformer Encoder/Decoder models.
        d_ff: int
            The dimension of the feedforward network.
        max_seq_length: int
            The maximum length of the input sequence.
        dropout: float
            The dropout rate.
        '''
        super(Transformer, self).__init__()
        # Initialize the word embeddings for the input sequence
        self.encoder_embedding = nn.Embedding(source_vocab_size, # number of words in the source vocabulary
                                              d_model, # dimension of the word embeddings
                                              )
        # Initialize the positional encoding for the input sequence
        self.decoder_embedding = nn.Embedding(target_vocab_size, # number of words in the target vocabulary
                                              d_model, # dimension of the word embeddings
                                              )
        # Initialize the positional encoding for the input sequence
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Initialize the Transformer encoder model
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # Initialize the Transformer decoder model
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Initialize the output linear layer
        self.fc = nn.Linear(d_model, target_vocab_size)
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        '''Generate the mask tensors for the input and output sequences.
        
        Parameters:
        ----------
        source: torch.Tensor
            The input sequence tensor with shape (batch_size, source_seq_len).
        target: torch.Tensor
            The output sequence tensor with shape (batch_size, target_seq_len).
            
        Returns:
        -------
        torch.Tensor
            The mask tensor for the input sequence with shape (batch_size, source_seq_len, source_seq_len).
        torch.Tensor
            The mask tensor for the output sequence with shape (batch_size, target_seq_len, target_seq_len).
        '''
        # Generate the mask tensor for the input sequence
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, source_seq_len)
        # Generate the mask tensor for the output sequence
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3) # (batch_size, 1, target_seq_len, 1)
        
        seq_length = target.size(1) # target_seq_len
        # Generate the lower triangular mask tensor for the output sequence
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        # Combine the target mask and the lower triangular mask
        target_mask = target_mask & nopeak_mask

        return source_mask, target_mask
    
    def forward(self, source, target):
        '''The forward pass of the Transformer model.
        
        Parameters:
        ----------
        source: torch.Tensor
            The input sequence tensor with shape (batch_size, source_seq_len).
        target: torch.Tensor
            The output sequence tensor with shape (batch_size, target_seq_len).
            
        Returns:
        -------
        torch.Tensor
            The output tensor with shape (batch_size, target_seq_len, target_vocab_size).
        '''
        # Compute the mask tensors for the input and output sequences
        source_mask, target_mask = self.generate_mask(source, target)
        # Embed the input sequence
        source_embedded = self.encoder_embedding(source)
        # Embed the output sequence
        target_embedded = self.decoder_embedding(target)
        # Add the positional encoding to the input and output sequences
        source_embedded = self.positional_encoding(source_embedded)
        target_embedded = self.positional_encoding(target_embedded)
        # Apply the dropout layer to both the input and output sequences
        source_embedded = self.dropout(source_embedded)
        target_embedded = self.dropout(target_embedded)

        # Apply the Transformer encoder layers
        encoder_output = source_embedded # (batch_size, source_seq_len, d_model)
        for enc_layer in self.transformer_encoder:
            encoder_output = enc_layer(encoder_output, source_mask)

        # Apply the Transformer decoder layers
        decoder_output = target_embedded
        for dec_layer in self.transformer_decoder:
            decoder_output = dec_layer(decoder_output,
                                       encoder_output,
                                       source_mask,
                                       target_mask,
                                       )
            
        # Apply the output linear layer
        output = self.fc(decoder_output)

        return output