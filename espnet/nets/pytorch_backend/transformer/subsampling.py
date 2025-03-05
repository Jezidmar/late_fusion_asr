#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""
import math
import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
import torch.nn.functional as F




class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))








class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling

    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling."""
    if isinstance(ins, Conv1dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling2) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling3) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv2dSubsampling2) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1


class Conv1dSubsampling1(torch.nn.Module):
    """Convolutional 1D subsampling.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling1 object."""
        super(Conv1dSubsampling1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:1][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv1dSubsampling2(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling2 object."""
        super(Conv1dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:1][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv1dSubsampling3(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/3 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling3 object."""
        super(Conv1dSubsampling3, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:1][:, :, :-4:3]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


# Default approach


class Conv2dSubsamplingFUSED(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED, self).__init__()

        self.num_repr = num_repr
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(odim, odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )

        self.out = nn.Sequential(
            nn.Linear(self.num_repr * (odim * (((idim - 1) // 2 - 1) // 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)
        x = torch.cat(x, dim=-1)  # (4, b, c, t, f)

        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Expert(nn.Module):
    def __init__(self, input_dim, out):
        super(Expert, self).__init__()

        self.fc = nn.Linear(input_dim, out)

    def forward(self, x):
        return self.fc(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_repr):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_repr)

    def forward(self, x):
        return self.fc(x)


# First implementation of Smoe


class SMoE_MHA_enc_version_1(nn.Module):
    def __init__(self, input_dim, out=512, num_repr=4, top_k=2):
        super(SMoE_MHA_enc_version_1, self).__init__()
        # Input dim is here actually the embedding dimension.
        self.num_repr = num_repr
        self.top_k = top_k
        self.out = out
        # Input to gating network is: embedding_dim* 4 * 19

        self.experts = nn.ModuleList(
            [Expert(input_dim, self.out) for _ in range(num_repr)]
        )
        self.gating_network = GatingNetwork(
            self.out * self.num_repr * 19, self.num_repr
        )
        self.input_dim = input_dim

    def forward(self, x):
        batch_size, _, seq_length, channels = x[0].size()
        x_flat = torch.cat(x, dim=-1)  # So now I have batch_size,embedding,7,4*19
        x_flat = torch.permute(x_flat, (0, 2, 1, 3))
        x_flat = torch.reshape(
            x_flat, (batch_size, seq_length, self.out * self.num_repr * channels)
        )  # 4*19 is 4 times the output dimension of convolutional layer
        gating_scores = self.gating_network(x_flat)

        topk_scores, topk_indices = torch.topk(gating_scores, self.top_k, dim=-1)

        all_expert_outputs = [
            expert(torch.reshape(xx, (batch_size, seq_length, self.out * channels)))
            for expert, xx in zip(self.experts, x)
        ]
        all_expert_outputs = torch.stack(all_expert_outputs, dim=-1)

        selected_expert_outputs = torch.gather(
            all_expert_outputs,
            dim=3,
            index=topk_indices.unsqueeze(2).expand(-1, -1, self.out, -1),
        )
        weights = nn.functional.softmax(topk_scores, dim=-1).unsqueeze(
            dim=2
        )  # (batch_size*num_heads, seq_length, 2, 1)

        expert_outputs = (selected_expert_outputs * weights).sum(
            dim=-1
        )  # (batch_size, seq_length, head_dim)

        return expert_outputs


class Conv2dSubsamplingFUSED_15(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4, top_k=2):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_15, self).__init__()

        self.num_repr = num_repr
        self.top_k = top_k
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(odim, odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )

        self.out = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

        self.moe_network = SMoE_MHA_enc_version_1(
            input_dim=odim * (idim // 4 - 1),
            out=odim,
            num_repr=self.num_repr,
            top_k=self.top_k,
        )  # Hardcoded config for case of using 4 representations

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x_mask = x_mask  # All the inputs have same dimensions

        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]

        # Input shape: [ (b,c,t,f) , (b,c,t,f). ...  ]
        x = self.moe_network(x)
        x = self.out(x)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


# Second version of Smoe


class SMoE_MHA_enc_version_2(nn.Module):
    def __init__(self, input_dim, out=512, num_repr=4, top_k=4):
        super(SMoE_MHA_enc_version_2, self).__init__()
        # Input dim is here actually the embedding dimension.
        self.num_repr = num_repr
        self.top_k = top_k
        self.out = out
        # Input to gating network is: embedding_dim* 4 * 19

        self.experts = nn.ModuleList(
            [Expert(input_dim, self.out) for _ in range(self.num_repr)]
        )
        self.gating_network = GatingNetwork(
            self.out * self.num_repr * 19, self.num_repr
        )
        self.project = nn.Linear(self.top_k * self.out, self.out)
        self.input_dim = input_dim

    def forward(self, x):
        batch_size, _, seq_length, channels = x[0].size()
        x_flat = torch.cat(x, dim=-1)  # So now I have batch_size,embedding,seq_len,4*19
        x_flat = torch.permute(x_flat, (0, 2, 1, 3))
        x_flat = torch.reshape(
            x_flat, (batch_size, seq_length, self.out * self.num_repr * channels)
        )  # 4*19 is 4 times the output dimension of convolutional layer
        gating_scores = self.gating_network(x_flat)
        #routing_weights = F.gumbel_softmax(gating_scores, tau=1.0, hard=True)
        topk_scores, topk_indices = torch.topk(gating_scores, self.top_k, dim=-1)

        all_expert_outputs = [
            expert(torch.reshape(xx, (batch_size, seq_length, self.out * channels)))
            for expert, xx in zip(self.experts, x)
        ]
        all_expert_outputs = torch.stack(all_expert_outputs, dim=-1)

        selected_expert_outputs = torch.gather(
            all_expert_outputs,
            dim=3,
            index=topk_indices.unsqueeze(2).expand(-1, -1, self.out, -1),
        )
        weights = nn.functional.softmax(topk_scores, dim=-1).unsqueeze(
            dim=2
        )  # (batch_size*num_heads, seq_length, 2, 1)

        expert_outputs = (
            selected_expert_outputs * weights
        )  # (batch_size, seq_length, head_dim)
        selected_output = expert_outputs.permute(0, 1, 3, 2).reshape(
            batch_size, seq_length, self.top_k * self.out
        )
        x = self.project(selected_output)

        return x


class Conv2dSubsamplingFUSED_16(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4, top_k=2):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_16, self).__init__()

        self.num_repr = num_repr
        self.top_k = top_k
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(odim, odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )

        self.out = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

        self.moe_network = SMoE_MHA_enc_version_2(
            input_dim=odim * (idim // 4 - 1),
            out=odim,
            num_repr=self.num_repr,
            top_k=self.top_k,
        )  # Hardcoded config for case of using 4 representations

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x_mask = x_mask  # All the inputs have same dimensions

        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]

        # Input shape: [ (b,c,t,f) , (b,c,t,f). ...  ]
        x = self.moe_network(x)
        x = self.out(x)
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsamplingFUSED_17(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_17, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.num_repr = num_repr
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress = nn.ModuleList(
            [nn.Linear(19 * self.odim, self.odim) for _ in range(self.num_repr)]
        )

        self.out = nn.Sequential(
            nn.Linear(self.num_repr * self.odim, self.odim),
            pos_enc
            if pos_enc is not None
            else PositionalEncoding(self.odim, self.dropout_rate),
        )

        self.idim = idim

    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions

        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]

        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)

        x = torch.cat(x, dim=-1)  # (b, t', num_repr * odim)
        # Project from self.num_repr * self.odim to self.odim
        x = self.out(x)

        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsamplingFUSED_18(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_18, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(19 * self.odim, self.odim),
                    pos_enc
                    if pos_enc is not None
                    else PositionalEncoding(self.odim, self.dropout_rate),
                )
                for _ in range(self.num_repr)
            ]
        )

        self.out = nn.Linear(self.num_repr * self.odim, self.odim)

    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions

        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]

        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)

        # Concatenate
        pos_emb = x[0][1]
        x = [xx[0] for xx in x]
        # Concatenate
        x = torch.cat(x, dim=-1)  # (b, t', num_repr * odim)
        # Project from self.num_repr * self.odim to self.odim
        x = self.out(x)

        if x_mask is None:
            return (x, pos_emb), None
        return (x, pos_emb), x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsamplingFUSED_19(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4, num_heads=4):
        super(Conv2dSubsamplingFUSED_19, self).__init__()
        self.input_dim = idim
        self.num_heads = num_heads
        self.num_repr = num_repr
        self.head_dim = odim // num_heads
        self.odim = odim
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )

        # Linear layers for each representation
        self.linear_per_repr = nn.ModuleList(
            [nn.Linear(19 * self.odim, self.odim) for _ in range(self.num_repr)]
        )

        # Compression after concatenating heads from corresponding representations
        self.compress_concat = nn.Linear(self.num_repr * self.head_dim, self.head_dim)

        # Final compression to output_dim
        self.final_compress = nn.Linear(self.num_heads * self.head_dim, self.odim)
        self.out = nn.Sequential(
            nn.Linear(self.num_heads * self.head_dim, self.odim),
            pos_enc
            if pos_enc is not None
            else PositionalEncoding(self.odim, self.dropout_rate),
        )

    def forward(self, x, x_mask):
        # x_list is a list of tensors, each with shape (batch_size, time, input_dim)
        assert (
            len(x) == self.num_repr
        ), "Number of input tensors must match num_representations"

        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        # Step 1 & 2: Separate tensor into heads and process each representation
        batch_size, time, _ = x[0].size()
        multi_head = []
        for i, xx in enumerate(x):
            # Apply linear transformation
            transformed = self.linear_per_repr[i](xx)
            # Reshape to (batch_size, time, num_heads, head_dim)
            reshaped = transformed.view(
                batch_size, time, self.num_heads, self.head_dim
            ).permute(0, 2, 1, 3)
            # Permute to (batch_size, num_heads, time, head_dim)
            multi_head.append(reshaped)

        # Step 3: Concatenate heads from corresponding representations
        # Shape: (batch_size, num_heads, time, num_representations * head_dim)
        x = torch.cat(multi_head, dim=-1)

        # Step 4: Compress concatenated heads
        # Shape: (batch_size, num_heads, time, head_dim)
        x = self.compress_concat(x)

        # Step 5: Merge heads
        # Shape: (batch_size, time, num_heads * head_dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, time, -1)

        # Step 6: Final compression to odim
        # Shape: (batch_size, time, output_dim)
        x = self.out(x)

        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class Expert_ape(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Expert_ape, self).__init__()
        self.fc1 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        return self.fc1(x)


class GatingNetwork_ape(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork_ape, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return self.fc(x)


class SMoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=8, top_k=2):
        super(SMoE, self).__init__()
        self.experts = nn.ModuleList(
            [Expert_ape(input_dim, output_dim) for _ in range(num_experts)]
        )
        self.gating_network = GatingNetwork_ape(input_dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_scores = self.gating_network(x)
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Compute expert outputs for all experts in parallel
        all_expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=2
        )  # (batch_size, seq_len, num_experts, input_dim)
        # Select top-k expert outputs
        selected_expert_outputs = all_expert_outputs.gather(
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, all_expert_outputs.size(-1)),
        )  # (batch_size, seq_len, 2, input_dim)

        # Calculate weights and apply them
        weights = F.softmax(topk_scores, dim=-1).unsqueeze(
            -1
        )  # (batch_size, seq_len, 2, 1)
        expert_outputs = (selected_expert_outputs * weights).sum(
            dim=2
        )  # (batch_size, seq_len, input_dim)

        return expert_outputs


# Version 20


class Conv2dSubsamplingFUSED_20(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(
        self, idim, odim, dropout_rate, pos_enc=None, num_repr=4, num_experts=8, top_k=2
    ):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_20, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.num_experts = num_experts
        self.top_k = top_k
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(19 * self.odim, self.odim),
                    pos_enc
                    if pos_enc is not None
                    else PositionalEncoding(self.odim, self.dropout_rate),
                )
                for _ in range(self.num_repr)
            ]
        )

        self.out = SMoE(
            self.num_repr * self.odim,
            self.odim,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions

        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]

        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)

        pos_emb = x[0][1]
        x = [xx[0] for xx in x]

        # Concatenate
        x = torch.cat(x, dim=-1)  # (b, t', num_repr * odim)
        # Project from self.num_repr * self.odim to self.odim
        x = self.out(x)

        if x_mask is None:
            return (x, pos_emb), None
        return (x, pos_emb), x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsamplingFUSED_21_Timo_Lohr(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length) with multi-channel input."""
    
    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4,weights=[0.6,0.2,0.1,0.1]):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_21_Timo_Lohr, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.weights = weights
        self.convs = nn.Sequential(
            nn.Conv2d(self.num_repr, self.odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(self.odim, self.odim, 3, 2),
            nn.ReLU(),
        )
        
        self.out = nn.Sequential(
            nn.Linear(self.odim * (((idim - 1) // 2 - 1) // 2), self.odim),
            pos_enc if pos_enc is not None else PositionalEncoding(self.odim, self.dropout_rate),
        )
        
        self.idim = idim

    def forward(self, x, x_mask):
        """
        Subsample x.
        
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        # Stack inputs along the channel dimension
        x = [xx*weight for xx,weight in zip(x,self.weights)]
        x = torch.stack(x, dim=1)  # (batch, num_repr, time, idim)
        
        # Apply convolutional layers
        x = self.convs(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item."""
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsamplingFUSED_22_Mha(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None,num_repr=4,num_heads=16):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_22_Mha, self).__init__()
        self.odim = odim
        self.num_repr=num_repr
        self.dropout_rate=dropout_rate
        self.num_repr=num_repr
        self.num_heads = num_heads
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress=nn.ModuleList(
            [
            nn.Sequential(    
                nn.Linear(19*self.odim,self.odim),
                pos_enc if pos_enc is not None else PositionalEncoding(self.odim, self.dropout_rate),
            )
                for _ in range(self.num_repr)
            ]
        )
        self.norm = nn.LayerNorm(self.num_repr * self.odim)
        self.attend = nn.MultiheadAttention(embed_dim=self.num_repr*self.odim, num_heads=self.num_heads)
        self.out = nn.Linear(self.num_repr * self.odim, self.odim)
        

        
        self.idim = idim


    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions
        
        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]
        
        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys; sys.exit(1)
        

        pos_emb = x[0][1]
        x = [xx[0] for xx in x]
        # Concatenate
        x = torch.cat(x, dim=-1)  # (b, t', num_repr * odim)


        # Now, add the self attention module
        x = self.norm(x)
        x, _ = self.attend(x, x, x)
        # Project from self.num_repr * self.odim to self.odim
        x = self.out(x)
        
        if x_mask is None:
            return (x,pos_emb), None
        return (x,pos_emb), x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


# IFCNN for pre-fusion of images

class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

    
class IFCNN(nn.Module):
    def __init__(self, fuse_scheme=0):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme # MAX, MEAN, SUM
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=0, bias=False)


    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, *tensors):
        # Feature extraction
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)
        outs = self.operate(self.conv2, outs)
        # Feature fusion
        if self.fuse_scheme == 0: # MAX
            out = self.tensor_max(outs)
        elif self.fuse_scheme == 1: # SUM
            out = self.tensor_sum(outs)
        elif self.fuse_scheme == 2: # MEAN
            out = self.tensor_mean(outs)
        else: # Default: MAX
            out = self.tensor_max(outs)
        
        # Feature reconstruction
        out = self.conv3(out)
        out = self.conv4(out)
        return out

class Conv2dSubsamplingFUSED_23(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsamplingFUSED_23, self).__init__()

        self.fuse_images = IFCNN(fuse_scheme=0)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = [xx.unsqueeze(1) for xx in x]
        x = self.fuse_images(*x)
        #x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class ImageCompressor(nn.Module):
    def __init__(self, input_width, output_width):
        super(ImageCompressor, self).__init__()
        self.input_width = input_width
        self.output_width = output_width

    def forward(self, x):
        # x shape: (batch_size, height, width)
        batch_size, height, width = x.shape
        
        # Add channel dimension and compress
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
        x_compressed = F.interpolate(
            x,
            size=(height, self.output_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Remove channel dimension
        return x_compressed.squeeze(1)
    
class FusionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(FusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 2, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (seq_length, batch_size, embed_dim)
        # Self-Attention with Residual Connection and LayerNorm
        src2, _ = self.self_attn(x, x, x)
        x = x + self.dropout(src2)
        x = self.norm1(x)

        # Feed-Forward Network with Residual Connection and LayerNorm
        src2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(src2)
        x = self.norm2(x)
        return x
    
    
class Conv2dSubsamplingFUSED_24(nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None,num_repr=4):
        super(Conv2dSubsamplingFUSED_24, self).__init__()
        self.idim = idim
        self.num_repr = num_repr
        self.odim = odim
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )

        self.compressor = ImageCompressor(self.num_repr*self.idim,self.idim)
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )        
    
    def forward(self, x, x_mask):
        # Fuse the images

        # just 
        x = torch.cat(x,dim=2)
        fused = self.compressor(x)
        x = fused.unsqueeze(1)  # Add batch dimension
        x = self.conv(x)        # x shape: (1, odim, T, 1)
        
        # Flatten the features
        b, c, t, f = x.size()
        x = x.view(b, t, c * f)  # Flatten c and f (since f is 1)
        x = self.out(x)          # Apply linear layer and positional encoding
        
        if x_mask is None:
            return x, None
        # Adjust x_mask according to the subsampling
        x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        return x, x_mask
    
    def __getitem__(self, key):
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsamplingFUSED_25(nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None,num_repr=4,num_fusion_layers = 6,num_heads = 16):
        super(Conv2dSubsamplingFUSED_25, self).__init__()
        self.idim = idim
        self.num_repr = num_repr
        self.odim = odim
        self.dropout_rate = dropout_rate
        self.num_fusion_layers = num_fusion_layers
        self.num_heads = num_heads
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress=nn.ModuleList(
            [
            nn.Sequential(    
                nn.Linear(19*self.odim,self.odim),
                pos_enc if pos_enc is not None else PositionalEncoding(self.odim, self.dropout_rate),
            )
                for _ in range(self.num_repr)
            ]
        )




        self.blocks = nn.ModuleList(
            [   
                FusionBlock(embed_dim=self.num_repr * self.odim, num_heads=self.num_heads, dropout=self.dropout_rate)
                for _ in range(self.num_fusion_layers)
            ]
        )


        self.out = nn.Linear(self.num_repr *self.odim, self.odim)
     
    
    def forward(self, x, x_mask):
        # Fuse the images

        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]
        pos_emb = x[0][1]

        x = [xx[0] for xx in x] # exclude positional encoding
        # Concatenate
        x = torch.cat(x, dim=-1) 


        for block in self.blocks:
            x = block(x)
        
        x = self.out(x)
        
        if x_mask is None:
            return (x,pos_emb), None
        # Adjust x_mask according to the subsampling
        x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        return (x,pos_emb), x_mask
    
    def __getitem__(self, key):
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]






class Conv2dSubsampling1(torch.nn.Module):
    """Similar to Conv2dSubsampling module, but without any subsampling performed.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling1 object."""
        super(Conv2dSubsampling1, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (idim - 4), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Pass x through 2 Conv2d layers without subsampling.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim).
                where time' = time - 4.
            torch.Tensor: Subsampled mask (#batch, 1, time').
                where time' = time - 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-4]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 1) // 2 - 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsamplingFUSED_28_for_perceptual(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_28_for_perceptual, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(19 * self.odim, self.odim),
                    pos_enc
                    if pos_enc is not None
                    else PositionalEncoding(self.odim, self.dropout_rate)
                )
                for _ in range(self.num_repr)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.odim)  # This creates an instance of LayerNorm
                for _ in range(self.num_repr)  # This repeats it for the number of representations
            ]
        )




    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions

        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]

        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)

        # Concatenate
        #pos_emb = x[0][1]
        # Concatenate
        # Project from self.num_repr * self.odim to self.odim

        x_out = [(norm(xx[0]),xx[1]) for xx,norm in zip(x,self.norms)]
            
        if x_mask is None:
            return x_out, None
        return x_out, x_mask[:, :, :-2:2][:, :, :-2:2]
    


class ImprovedCrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attention_a = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.self_attention_v = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.modal_attention_a = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        self.modal_attention_v = nn.MultiheadAttention(embed_dim, num_heads,batch_first=True)
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, h_a, h_v):
        # Self-attention for audio
        h_a_self,_ = self.self_attention_a(h_a, h_a, h_a)
        h_a = h_a + h_a_self
        
        # Self-attention for visual
        h_v_self,_ = self.self_attention_v(h_v, h_v, h_v)
        h_v = h_v + h_v_self
        
        # Cross-modal attention
        q_a, k_v, v_v = h_a, h_v, h_v
        h_av,_ = self.modal_attention_a(q_a, k_v, v_v)
        h_a = h_a + h_av
        
        q_v, k_a, v_a = h_v, h_a, h_a
        h_va,_ = self.modal_attention_v(q_v, k_a, v_a)
        h_v = h_v + h_va
        
        # Final normalization
        h_a_prime = self.norm(h_a)
        h_v_prime = self.norm(h_v)
        
        # Combine modalities
        h_av = h_a_prime + h_v_prime
        
        return h_a_prime, h_v_prime, h_av

class Conv2dSubsamplingFUSED_26(nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None,num_repr=4,num_heads = 8):
        super(Conv2dSubsamplingFUSED_26, self).__init__()
        self.idim = idim
        self.num_repr = num_repr
        self.odim = odim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.norm_1 = nn.LayerNorm(self.odim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.odim, num_heads=self.num_heads,dropout=0.2,batch_first=True)
            for _ in range(self.num_repr - 2)
        ])
        self.compress=nn.ModuleList(
            [
            nn.Sequential(    
                nn.Linear(19*self.odim,self.odim),
                pos_enc if pos_enc is not None else PositionalEncoding(self.odim, self.dropout_rate)
            )
                for _ in range(self.num_repr)
            ]
        )
        self.final_attend = ImprovedCrossAttentionModule(self.odim,self.num_heads) # hardcoded to 8 
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.odim)  # This creates an instance of LayerNorm
                for _ in range(self.num_repr)  # This repeats it for the number of representations
            ]
        )

        self.out = PositionwiseFeedForward(self.odim, self.num_repr*self.odim,self.dropout_rate)
     
    
    def forward(self, x, x_mask):
        # Fuse the images

        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]
        pos_emb = x[0][1]

        x = [xx[0] for xx in x] # exclude positional encoding
        x = [norm(xx) for norm,xx in zip(self.norms,x)]
        first_rep = x[0]
        compress_repr= x[1:]
        # We need dropout and residual connections, of course
        number_of_reps_to_compress = len(compress_repr)
        ref_repr = compress_repr[1]
        for i in range(number_of_reps_to_compress-1):
            second_rep = ref_repr
            third_rep = compress_repr[i+1]
            ref_repr = self.cross_attn[i](second_rep,third_rep,third_rep)[0] + self.dropout(ref_repr) # Residual connection

        # Now fuse repr repr and first repr
        ref_repr = self.norm_1(ref_repr)
        _, _, x = self.final_attend(first_rep,ref_repr)

        x = self.out(x)
        
        if x_mask is None:
            return (x,pos_emb), None
        # Adjust x_mask according to the subsampling
        x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        return (x,pos_emb), x_mask
    
    def __getitem__(self, key):
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
    


class Conv2dSubsamplingFUSED_27(nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None,num_repr=4,num_heads = 8):
        super(Conv2dSubsamplingFUSED_27, self).__init__()
        self.idim = idim
        self.num_repr = num_repr
        self.odim = odim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.norm_1 = nn.LayerNorm(self.odim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.odim, num_heads=self.num_heads,dropout=0.2,batch_first=True)
            for _ in range(self.num_repr - 2)
        ])
        self.ffns = nn.ModuleList([
            PositionwiseFeedForward(self.odim, self.num_repr*self.odim,self.dropout_rate)
            for _ in range(self.num_repr - 2)
        ])
        self.layerNorms = nn.ModuleList([
            nn.LayerNorm(self.odim)
            for _ in range(self.num_repr - 2)
        ])


        self.compress=nn.ModuleList(
            [
            nn.Sequential(    
                nn.Linear(19*self.odim,self.odim),
                pos_enc if pos_enc is not None else PositionalEncoding(self.odim, self.dropout_rate)
            )
                for _ in range(self.num_repr)
            ]
        )
        self.final_attend = ImprovedCrossAttentionModule(self.odim,self.num_heads) # hardcoded to 8 
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.odim)  # This creates an instance of LayerNorm
                for _ in range(self.num_repr)  # This repeats it for the number of representations
            ]
        )

        self.out = PositionwiseFeedForward(self.odim, self.num_repr*self.odim,self.dropout_rate)
     
    
    def forward(self, x, x_mask):
        # Fuse the images

        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]
        pos_emb = x[0][1]

        x = [xx[0] for xx in x] # exclude positional encoding
        x = [norm(xx) for norm,xx in zip(self.norms,x)]

        first_rep = x[0]
        compress_repr= x[1:]
        # We need dropout and residual connections, of course
        number_of_reps_to_compress = len(compress_repr)
        ref_repr = compress_repr[1]
        for i in range(number_of_reps_to_compress-1):
            second_rep = ref_repr
            third_rep = compress_repr[i+1]
            ref_repr = self.cross_attn[i](second_rep,third_rep,third_rep)[0]  
            ref_repr = self.ffns[i](ref_repr) + self.dropout(ref_repr) # Residual connection
            ref_repr = self.layerNorms[i](ref_repr)

        # Now fuse repr repr and first repr
        first_rep = self.norm_1(first_rep)
        _, _, x = self.final_attend(first_rep,ref_repr)

        x = self.out(x)
        
        if x_mask is None:
            return (x,pos_emb), None
        # Adjust x_mask according to the subsampling
        x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        return (x,pos_emb), x_mask
    
    def __getitem__(self, key):
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
    


class Conv2dSubsamplingFUSED_28_for_perceptual_v2(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_28_for_perceptual_v2, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.shared_conv = nn.Conv2d(self.odim,self.odim,3,2)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    self.shared_conv,
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(19 * self.odim, self.odim),
                    pos_enc
                    if pos_enc is not None
                    else PositionalEncoding(self.odim, self.dropout_rate)
                )
                for _ in range(self.num_repr)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.odim)  # This creates an instance of LayerNorm
                for _ in range(self.num_repr)  # This repeats it for the number of representations
            ]
        )




    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions

        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]

        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)

        # Concatenate
        #pos_emb = x[0][1]
        # Concatenate
        # Project from self.num_repr * self.odim to self.odim

        x_out = [(norm(xx[0]),xx[1]) for xx,norm in zip(x,self.norms)]
            
        if x_mask is None:
            return x_out, None
        return x_out, x_mask[:, :, :-2:2][:, :, :-2:2]
    

class Conv2dSubsamplingFUSED_28_for_perceptual_v3(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_28_for_perceptual_v3, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    nn.Conv2d(self.odim, self.odim, 3, 2),
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.shared_linear = nn.Linear(19* self.odim, self.odim)
        self.compress = nn.ModuleList(
            [
                nn.Sequential(
                    self.shared_linear,
                    pos_enc
                    if pos_enc is not None
                    else PositionalEncoding(self.odim, self.dropout_rate)
                )
                for _ in range(self.num_repr)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(self.odim)  # This creates an instance of LayerNorm
                for _ in range(self.num_repr)  # This repeats it for the number of representations
            ]
        )




    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions

        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]

        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)

        # Concatenate
        #pos_emb = x[0][1]
        # Concatenate
        # Project from self.num_repr * self.odim to self.odim

        x_out = [(norm(xx[0]),xx[1]) for xx,norm in zip(x,self.norms)]
            
        if x_mask is None:
            return x_out, None
        return x_out, x_mask[:, :, :-2:2][:, :, :-2:2]
    

class Conv2dSubsamplingFUSED_29(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, num_repr=4):
        """Construct an Conv2dSubsamplingFUSED object."""
        super(Conv2dSubsamplingFUSED_29, self).__init__()
        self.odim = odim
        self.num_repr = num_repr
        self.dropout_rate = dropout_rate
        self.shared_conv = nn.Conv2d(self.odim,self.odim,3,2)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, self.odim, 3, 2),
                    nn.ReLU(),
                    self.shared_conv,
                    nn.ReLU(),
                )
                for _ in range(self.num_repr)
            ]
        )
        self.compress = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(19 * self.odim, self.odim),
                    pos_enc
                    if pos_enc is not None
                    else PositionalEncoding(self.odim, self.dropout_rate),
                )
                for _ in range(self.num_repr)
            ]
        )

        self.out = nn.Linear(self.num_repr * self.odim, self.odim)

    def forward(self, x, x_mask):
        """Subsample x.
        Args:
            x (List[torch.Tensor]): List of input tensors (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        len_before_subs = len(x)
        x_mask = x_mask  # All the inputs have same dimensions

        # Apply convolutional layers
        x = [xx.unsqueeze(1) for xx in x]  # [(b, 1, t, f), (b, 1, t, f), ...]
        x = [conv(xx) for conv, xx in zip(self.convs, x)]
        bs, emb, t, comp = x[0].size()
        x = [xx.permute(0, 2, 3, 1).reshape(bs, t, -1) for xx in x]
        x = [compr(xx) for compr, xx in zip(self.compress, x)]

        len_after_subs = len(x)
        if len_before_subs != len_after_subs:
            print("Wrong number of representations used")
            import sys

            sys.exit(1)

        # Concatenate
        pos_emb = x[0][1]
        x = [xx[0] for xx in x]
        # Concatenate
        x = torch.cat(x, dim=-1)  # (b, t', num_repr * odim)
        # Project from self.num_repr * self.odim to self.odim
        x = self.out(x)

        if x_mask is None:
            return (x, pos_emb), None
        return (x, pos_emb), x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]