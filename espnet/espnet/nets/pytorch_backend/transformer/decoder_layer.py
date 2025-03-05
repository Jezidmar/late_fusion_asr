#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        sequential_attn (bool): computes attn first on pre_x then on x,
                thereby attending to two sources in sequence.


    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        sequential_attn=None,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sequential_attn = sequential_attn
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        if sequential_attn is not None:
            self.norm4 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
            if sequential_attn is not None:
                self.concat_linear3 = nn.Linear(size + size, size)

    def forward(
        self,
        tgt,
        tgt_mask,
        memory,
        memory_mask,
        cache=None,
        pre_memory=None,
        pre_memory_mask=None,
    ):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).
            pre_memory (torch.Tensor): Encoded memory (#batch, maxlen_in, size).
            pre_memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        if self.sequential_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm4(x)
            if self.concat_after:
                x_concat = torch.cat(
                    (
                        x,
                        self.sequential_attn(
                            x, pre_memory, pre_memory, pre_memory_mask
                        ),
                    ),
                    dim=-1,
                )
                x = residual + self.concat_linear3(x_concat)
            else:
                x = residual + self.dropout(
                    self.sequential_attn(x, pre_memory, pre_memory, pre_memory_mask)
                )
            if not self.normalize_before:
                x = self.norm4(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pre_memory is not None:
            return x, tgt_mask, memory, memory_mask, None, pre_memory, pre_memory_mask
        return x, tgt_mask, memory, memory_mask



class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim,activation='relu'):
        super(Expert, self).__init__()
       
        if activation=='relu':            
            self.activation = nn.ReLU()
        if activation=='gelu':
            self.activation = nn.GELU()
        if activation=='glu':
            self.activation=GEGLU()
        
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



class SMoE_combined(nn.Module):
    "Combined mixture of experts with local and global extractor branches"
    def __init__(self, input_dim, hidden_dim, num_experts,top_k, num_heads,activation):
        super(SMoE_combined, self).__init__()
        # Input dim is here actually the embedding dimension.
        self.h = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        #Hardcode activation function for now.
        self.Global_extractor=SMoE(input_dim, input_dim*4, self.num_experts,self.top_k,'relu')
        self.Local_extractor=SMoE_MHA(input_dim, input_dim*4,self.num_experts,self.top_k, self.h,'relu')
    def forward(self, x):        
        x_global,stats_global = self.Global_extractor(x)
        x_local,stats_local = self.Local_extractor(x)
        
        x = x_global+x_local  #For first trial, go without router.
        return x, (stats_local,stats_global) 
       



class DecoderLayerMOE(nn.Module):
    """Single decoder layer module."""

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        sequential_attn=None,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayerMOE, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.sequential_attn = sequential_attn
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        if sequential_attn is not None:
            self.norm4 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.feed_forward = feed_forward  # Using SMoE block with num_experts experts
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
            if sequential_attn is not None:
                self.concat_linear3 = nn.Linear(size + size, size)

    def forward(
        self,
        tgt,
        tgt_mask,
        memory,
        memory_mask,
        cache=None,
        pre_memory=None,
        pre_memory_mask=None,
    ):
        """Compute decoded features."""
        #print(f"tgt:{tgt}, tgt_mask:{tgt_mask}, memory:{memory},memory_mask:{memory_mask}")
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        if self.sequential_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.norm4(x)
            if self.concat_after:
                x_concat = torch.cat(
                    (
                        x,
                        self.sequential_attn(
                            x, pre_memory, pre_memory, pre_memory_mask
                        ),
                    ),
                    dim=-1,
                )
                x = residual + self.concat_linear3(x_concat)
            else:
                x = residual + self.dropout(
                    self.sequential_attn(x, pre_memory, pre_memory, pre_memory_mask)
                )
            if not self.normalize_before:
                x = self.norm4(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        out_ffn = self.feed_forward(x)
        x = residual + self.dropout(out_ffn[0])
        
        
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pre_memory is not None:
            return x, tgt_mask, memory, memory_mask, None, pre_memory, pre_memory_mask,out_ffn[1]
        return x, tgt_mask, memory, memory_mask,out_ffn[1]


