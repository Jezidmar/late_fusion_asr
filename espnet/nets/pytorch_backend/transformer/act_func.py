import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)
