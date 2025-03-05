#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
from torch import nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x_q, x, x, mask)
            )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim,activation='relu'):
        super(Expert, self).__init__()
       
        if activation=='relu':            
            self.activation = nn.ReLU()
        if activation=='gelu':
            self.activation = nn.GELU()

        
        if activation=='glu':
            self.fc2 = nn.Linear(hidden_dim//2, input_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return self.fc(x)



#My implementation of Sparse Mixture of experts layer

class SMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts,top_k,activation):
        super(SMoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim,activation) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts
    def forward(self, x):
        device = x.device
        gate_scores = self.gating_network(x)
        softmax_scores = F.softmax(gate_scores,dim=-1)
        topk_scores, topk_indices = torch.topk(softmax_scores, self.top_k, dim=-1)

        batch_size, seq_len, _ = x.size()
        
        # Compute expert outputs for all experts in parallel
        all_expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # (batch_size, seq_len, num_experts, input_dim)
        
        # Select top-k expert outputs
        selected_expert_outputs = all_expert_outputs.gather(self.top_k, topk_indices.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))  # (batch_size, seq_len, top_k, input_dim)
        
        # Calculate weights and apply them
        weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # (batch_size, seq_len, 2, 1)
        expert_outputs = (selected_expert_outputs * weights).sum(dim=2)  # (batch_size, seq_len, input_dim)
        
        return expert_outputs, (topk_scores, topk_indices, gate_scores)



#My implementation of Multi Headed Mixture of experts layer

class SMoE_MHA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts,top_k, num_heads,activation):
        super(SMoE_MHA, self).__init__()
        # Input dim is here actually the embedding dimension.
        self.h = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(input_dim // self.h, hidden_dim,activation) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(input_dim // self.h, self.num_experts)
        self.pre_split_projection = nn.Linear(input_dim, input_dim)
        self.align = nn.Linear(input_dim, input_dim)
        self.embedding_dim = input_dim
    
    def forward(self, x):
        # Hardcode num_heads

        # x in shape (Batch, Time, Emb)
        batch_size, seq_length, embedding_dim = x.size()
        assert embedding_dim == self.embedding_dim, "Embedding dimension mismatch"

        # Calculate new embedding dimension per head
        head_dim = embedding_dim // self.h
        assert head_dim * self.h == embedding_dim, "Embedding dimension must be divisible by number of heads"

        # Step 0: Project
        x = self.pre_split_projection(x)
        
        # Step 1: Reshape to [batch_size, num_heads, seq_length, head_dim]
        x = x.view(batch_size, seq_length, self.h, head_dim).permute(0, 2, 1, 3)
        
        # Step 2: Flatten the last two dimensions for gating network input
        x_flat = x.contiguous().view(batch_size * self.h, seq_length, head_dim)
        
        # Step 3: Gating network
        gate_scores = self.gating_network(x_flat)
        softmax_scores = F.softmax(gate_scores,dim=-1)
        topk_scores, topk_indices = torch.topk(softmax_scores, self.top_k, dim=-1)

        # Step 4: Compute outputs for all experts
        all_expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=2)  # (batch_size*num_heads, seq_length, num_experts, head_dim)
        
        # Step 5: Select top-k expert outputs
        selected_expert_outputs = all_expert_outputs.gather(2, topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))  # (batch_size*num_heads, seq_length, 2, head_dim)
        
        # Step 6: Calculate weights and apply them
        weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1)  # (batch_size*num_heads, seq_length, 2, 1)
        expert_outputs = (selected_expert_outputs * weights).sum(dim=2)  # (batch_size*num_heads, seq_length, head_dim)
        
        # Step 7: Apply residual connection
        x_flat = x_flat + expert_outputs

        # Step 8: Reshape back to original dimensions
        x = x_flat.view(batch_size, self.h, seq_length, head_dim).permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size, seq_length, embedding_dim)
        
        # Step 9: Apply FFN to align mis-paired values
        x = self.align(x)
        
        return x, (topk_scores, topk_indices, gate_scores)







class EncoderLayerMOE(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayerMOE, self).__init__()
        self.self_attn = self_attn
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.num_experts=8
        self.top_k=2
        self.h=4
        self.Global_extractor=SMoE(size, size*4, self.num_experts,self.top_k,'relu')
        self.Local_extractor=SMoE_MHA(size, size*4,self.num_experts,self.top_k, self.h,'relu')

    def forward(self, x, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        x,_= self.Local_extractor(x)
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x_q, x, x, mask)
            )
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x,_ = self.Global_extractor(x)
        x = residual + stoch_layer_coeff * self.dropout(x)
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask
