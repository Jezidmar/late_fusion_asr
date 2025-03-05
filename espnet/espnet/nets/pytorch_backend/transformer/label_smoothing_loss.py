#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Label smoothing module."""

import torch
from torch import nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """Label-smoothing loss.

    :param int size: the number of class
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length

    def forward(self, x, target):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        return kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom


class LabelSmoothingLossMoE(nn.Module):
    """Label-smoothing loss combined with Router Load Balancing loss.

    :param int size: the number of classes
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    :param float load_loss_coeff: coefficient for Load Balancing loss
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
        num_experts=8,
        load_balance_loss_coef=0,
        z_loss_coeff=0
    ):
        """Construct a CombinedLabelSmoothingAndLoadBalancingLoss object."""
        super(LabelSmoothingLossMoE, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length
        if load_balance_loss_coef:
            self.load_loss_coeff = load_balance_loss_coef
        else:
            self.load_loss_coeff = 0
        if z_loss_coeff:
            self.z_loss_coeff = z_loss_coeff
        else:
            self.z_loss_coeff = 0

        self.num_experts=num_experts
    def forward(self, x, router_out, target):
        """Compute the combined loss.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :param torch.Tensor router_logits: logits for Router Load Balancing loss
        :return: scalar float value
        :rtype: torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        
        # Label Smoothing Loss
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        kl = self.criterion(F.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        label_smoothing_loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        
        # Calculate Load loss now.
        
        topk_values, topk_indices,gate_logits = router_out
        #print(router_logits.shape) #Router logits [19,50] for example
        batch_size, seq_len, k = topk_values.shape
        num_elements = batch_size * seq_len  # |X|
        
        # Initialize the load for each expert
        expert_loads = torch.zeros(self.num_experts, device=topk_values.device)
        
        # Calculate the load for each expert
        for i in range(self.num_experts):
            expert_loads[i] = (topk_indices == i).sum().item()
        
        # Calculate the logits sum for each expert
        logits_sum = torch.zeros(self.num_experts, device=topk_values.device)
        for p in range(self.num_experts):
            mask = topk_indices == p  # Mask to select the logits corresponding to expert p
            logits_sum[p] = (topk_values * mask.float()).sum().item()

        load_loss = (self.num_experts / num_elements) * (expert_loads * logits_sum).sum()
        
        # Calculate Z - loss now.
        
        router_z_loss = torch.logsumexp(gate_logits, dim = -1)
        router_z_loss = torch.square(router_z_loss)            
        router_z_loss = router_z_loss.mean()        

        # Combine the losses
        total_loss = label_smoothing_loss + self.load_loss_coeff * load_loss + self.z_loss_coeff * router_z_loss
        
        return total_loss



class LabelSmoothingLossMoECombined(nn.Module):
    """Label-smoothing loss combined with Router Load Balancing loss.

    :param int size: the number of classes
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    :param float load_balance_loss_coef: coefficient for Load Balancing loss
    :param float z_loss_coeff: coefficient for Z loss
    :param int num_experts: number of experts for load balancing
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
        num_experts=8,
        load_balance_loss_coef=0,
        z_loss_coeff=0
    ):
        """Construct a CombinedLabelSmoothingAndLoadBalancingLoss object."""
        super(LabelSmoothingLossMoECombined, self).__init__()
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.normalize_length = normalize_length
        self.criterion = criterion
        self.num_experts = num_experts
        self.load_balance_loss_coef = load_balance_loss_coef
        self.z_loss_coeff = z_loss_coeff
        self.confidence = 1.0 - smoothing

    def forward(self, x, router_out, target):
        """Compute the combined loss.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_idx (batch, seqlen)
        :param tuple router_out: tuple containing local and global router outputs
        :return: scalar float value
        :rtype: torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        
        # Label Smoothing Loss
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        kl = self.criterion(F.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        label_smoothing_loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        
        # Extract local and global router outputs
        topk_values_local, topk_indices_local, gate_logits_local = router_out[0]
        topk_values_global, topk_indices_global, gate_logits_global = router_out[1]

        # Calculate Load Loss
        expert_loads_local = torch.zeros(self.num_experts, device=topk_values_local.device)
        expert_loads_global = torch.zeros(self.num_experts, device=topk_values_global.device)
        logits_sum_local = torch.zeros(self.num_experts, device=topk_values_local.device)
        logits_sum_global = torch.zeros(self.num_experts, device=topk_values_global.device)

        for i in range(self.num_experts):
            expert_loads_local[i] = (topk_indices_local == i).sum()
            expert_loads_global[i] = (topk_indices_global == i).sum()
            logits_sum_local[i] = (topk_values_local[topk_indices_local == i]).sum()
            logits_sum_global[i] = (topk_values_global[topk_indices_global == i]).sum()

        num_elements_local = topk_values_local.numel() / topk_values_local.size(-1)
        num_elements_global = topk_values_global.numel() / topk_values_global.size(-1)
        
        load_loss_local = (self.num_experts / num_elements_local) * (expert_loads_local * logits_sum_local).sum()
        load_loss_global = (self.num_experts / num_elements_global) * (expert_loads_global * logits_sum_global).sum()

        # Calculate Z Loss
        router_z_loss_local = torch.square(torch.logsumexp(gate_logits_local, dim=-1)).mean()
        router_z_loss_global = torch.square(torch.logsumexp(gate_logits_global, dim=-1)).mean()

        # Combine the losses
        total_loss = label_smoothing_loss \
                     + self.load_balance_loss_coef * (load_loss_local + load_loss_global) / 2 \
                     + self.z_loss_coeff * (router_z_loss_local + router_z_loss_global) / 2
        
        return total_loss




class LabelSmoothingLossMoE_Dynamic(nn.Module):
    """Label-smoothing loss combined with Dynamic loss.

    :param int size: the number of classes
    :param int padding_idx: ignored class id
    :param float smoothing: smoothing rate (0.0 means the conventional CE)
    :param bool normalize_length: normalize loss by sequence length if True
    :param torch.nn.Module criterion: loss function to be smoothed
    :param float dynamic_loss_coef: coefficient for Dynamic loss
    """

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
        dynamic_loss_coef=1e-4
    ):
        """Construct a CombinedLabelSmoothingAndLoadBalancingLoss object."""
        super(LabelSmoothingLossMoE_Dynamic, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length
        self.dynamic_loss_coef = dynamic_loss_coef
        self.num_experts=8
        self.eps=1e-8
    def forward(self, x, router_out, target):
        """Compute the combined loss.

        :param torch.Tensor x: prediction (batch, seqlen, class)
        :param torch.Tensor target: target signal masked with self.padding_id (batch, seqlen)
        :param torch.Tensor router_logits: logits for Router Load Balancing loss
        :return: scalar float value
        :rtype: torch.Tensor
        """
        assert x.size(2) == self.size
        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        
        # Label Smoothing Loss
        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.size - 1))
            ignore = target == self.padding_idx  # (B,)
            total = len(target) - ignore.sum().item()
            target = target.masked_fill(ignore, 0)  # avoid -1 index
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        kl = self.criterion(F.log_softmax(x, dim=1), true_dist)
        denom = total if self.normalize_length else batch_size
        label_smoothing_loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
        
        # Router Z Loss
        _, _,gate_logits = router_out
        router_softmax = F.softmax(gate_logits,dim=-1)

        dynamic_loss = -torch.sum(router_softmax*torch.log(router_softmax+self.eps))
        total_loss = label_smoothing_loss + self.dynamic_loss_coef * dynamic_loss        

        return total_loss
