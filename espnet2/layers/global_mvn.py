from pathlib import Path
from typing import Tuple, Union, List
import torch.nn as nn
import numpy as np
import torch
from typeguard import check_argument_types
import yaml
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class GlobalMVN(AbsNormalize, InversibleInterface):
    """Apply global mean and variance normalization

    TODO(kamo): Make this class portable somehow

    Args:
        stats_file: npy file
        norm_means: Apply mean normalization
        norm_vars: Apply var normalization
        eps:
    """

    def __init__(
        self,
        stats_file: Union[Path, str],
        norm_means: bool = True,
        norm_vars: bool = True,
        eps: float = 1.0e-20,
    ):
        assert check_argument_types()
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps
        stats_file = Path(stats_file)

        self.stats_file = stats_file
        stats = np.load(stats_file)
        if isinstance(stats, np.ndarray):
            # Kaldi like stats
            count = stats[0].flatten()[-1]
            mean = stats[0, :-1] / count
            var = stats[1, :-1] / count - mean * mean
        else:
            # New style: Npz file
            count = stats["count"]
            sum_v = stats["sum"]
            sum_square_v = stats["sum_square"]
            mean = sum_v / count
            var = sum_square_v / count - mean * mean
        std = np.sqrt(np.maximum(var, eps))

        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        else:
            mean = torch.tensor(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std)
        else:
            std = torch.tensor(std).float()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def extra_repr(self):
        return (
            f"stats_file={self.stats_file}, "
            f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"
        )

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)
        """
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)
        mask = make_pad_mask(ilens, x, 1)

        # feat: (B, T, D)
        if norm_means:
            if x.requires_grad:
                x = x - self.mean
            else:
                x -= self.mean
        if x.requires_grad:
            x = x.masked_fill(mask, 0.0)
        else:
            x.masked_fill_(mask, 0.0)

        if norm_vars:
            x /= self.std

        return x, ilens

    def inverse(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)
        mask = make_pad_mask(ilens, x, 1)

        if x.requires_grad:
            x = x.masked_fill(mask, 0.0)
        else:
            x.masked_fill_(mask, 0.0)

        if norm_vars:
            x *= self.std

        # feat: (B, T, D)
        if norm_means:
            x += self.mean
            x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)
        return x, ilens


class FusedFeatureNormalizeGlobal(AbsNormalize):
    """Class for applying SpecAug and UtteranceMVN to fused features."""

    def __init__(
        self,
        norm_means: bool = True,
        norm_vars: bool = True,
        eps: float = 1.0e-20,
    ):
        super().__init__()

        config_path="conf/global_mvn_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        stats_files = config.get('stats_files', [])
        if not stats_files:
            raise ValueError("No stats files provided in the config.")
        
        # Create a GlobalMVN instance for each stats file
        self.global_mvns = nn.ModuleList([
            GlobalMVN(norm_means=norm_means, norm_vars=norm_vars, eps=eps, stats_file=str(stats_file))
            for stats_file in stats_files
        ])

    def forward(
        self, features: List[torch.Tensor], lengths: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        normalized_features = []
        for feature, global_mvn in zip(features, self.global_mvns):
            normalized_feature, _ = global_mvn(feature, lengths)
            normalized_features.append(normalized_feature)
        return normalized_features, lengths