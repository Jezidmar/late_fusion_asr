"""Time warp module."""

from typing import List

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list

DEFAULT_TIME_WARP_MODE = "bicubic"


def time_warp(x: torch.Tensor, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
    """Time warping using torch.interpolate.

    Args:
        x: (Batch, Time, Freq)
        window: time warp parameter
        mode: Interpolate mode
    """

    # bicubic supports 4D or more dimension tensor
    org_size = x.size()
    if x.dim() == 3:
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        x = x[:, None]

    t = x.shape[2]
    if t - window <= window:
        return x.view(*org_size)

    center = torch.randint(window, t - window, (1,))[0]
    warped = torch.randint(center - window, center + window, (1,))[0] + 1

    # left: (Batch, Channel, warped, Freq)
    # right: (Batch, Channel, time - warped, Freq)
    left = torch.nn.functional.interpolate(
        x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False
    )
    right = torch.nn.functional.interpolate(
        x[:, :, center:], (t - warped, x.shape[3]), mode=mode, align_corners=False
    )

    if x.requires_grad:
        x = torch.cat([left, right], dim=-2)
    else:
        x[:, :, :warped] = left
        x[:, :, warped:] = right

    return x.view(*org_size)


class TimeWarp(torch.nn.Module):
    """Time warping using torch.interpolate.

    Args:
        window: time warp parameter
        mode: Interpolate mode
    """

    def __init__(self, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
        super().__init__()
        self.window = window
        self.mode = mode

    def extra_repr(self):
        return f"window={self.window}, mode={self.mode}"

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            x: (Batch, Time, Freq)
            x_lengths: (Batch,)
        """

        if x_lengths is None or all(le == x_lengths[0] for le in x_lengths):
            # Note that applying same warping for each sample
            y = time_warp(x, window=self.window, mode=self.mode)
        else:
            # FIXME(kamo): I have no idea to batchify Timewarp
            ys = []
            for i in range(x.size(0)):
                _y = time_warp(
                    x[i][None, : x_lengths[i]],
                    window=self.window,
                    mode=self.mode,
                )[0]
                ys.append(_y)
            y = pad_list(ys, 0.0)

        return y, x_lengths


# Fused features
def time_warp_fused(
    x: torch.Tensor,
    mode: str = DEFAULT_TIME_WARP_MODE,
    center: torch.Tensor = None,
    warped: torch.Tensor = None,
    org_size: torch.Tensor = None,
    t: int = None,
):
    """Time warping using torch.interpolate.

    Args:
        x: (Batch, Time, Freq)
        mode: Interpolate mode
    """

    # bicubic supports 4D or more dimension tensor

    # left: (Batch, Channel, warped, Freq)
    # right: (Batch, Channel, time - warped, Freq)
    left = torch.nn.functional.interpolate(
        x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False
    )
    right = torch.nn.functional.interpolate(
        x[:, :, center:], (t - warped, x.shape[3]), mode=mode, align_corners=False
    )

    if x.requires_grad:
        x = torch.cat([left, right], dim=-2)
    else:
        x[:, :, :warped] = left
        x[:, :, warped:] = right

    return x.view(*org_size)


class TimeWarpFused(torch.nn.Module):
    """Time warping using torch.interpolate.

    Args:
        window: time warp parameter
        mode: Interpolate mode
    """

    def __init__(self, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
        super().__init__()
        self.window = window
        self.mode = mode

    def extra_repr(self):
        return f"window={self.window}, mode={self.mode}"

    def forward(self, x: List[torch.Tensor], x_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            x: List[(Batch, Time, Freq)]
            x_lengths: (Batch,)
        """
        out = []
        for feat in x:
            ys = []

            for i in range(feat.size(0)):
                tens = feat[i][None, : x_lengths[i]]
                org_size = tens.size()

                if tens.dim() == 3:
                    # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
                    tens = tens[:, None]

                t = tens.shape[2]
                if t - self.window <= self.window:
                    return tens.view(*org_size)

                # Compute random values for all features

                center = torch.randint(self.window, t - self.window, (1,))[0]
                warped = (
                    torch.randint(center - self.window, center + self.window, (1,))[0]
                    + 1
                )

                _y = time_warp_fused(
                    tens,
                    mode=self.mode,
                    center=center,
                    warped=warped,
                    org_size=org_size,
                    t=t,
                )[0]
                ys.append(_y)
                del tens, org_size, _y, center, warped, t

            y = pad_list(ys, 0.0)

            out.append(y)
        return out, x_lengths
