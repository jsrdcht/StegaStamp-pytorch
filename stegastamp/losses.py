from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib
try:
    lpips = importlib.import_module('lpips')
except Exception:
    lpips = None
try:
    kornia = importlib.import_module('kornia')
except Exception:
    kornia = None


class LPIPSLoss(nn.Module):
    def __init__(self, net: str = 'alex'):
        super().__init__()
        if lpips is None:
            self.model = None
        else:
            self.model = lpips.LPIPS(net=net)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return F.mse_loss(x, y)
        return self.model(x, y).mean()


def _rgb_to_yuv_fallback(img: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch RGB->YUV fallback when kornia is unavailable.
    Formula aligned with common convention used by kornia:
      Y = 0.299 R + 0.587 G + 0.114 B
      U = -0.14713 R - 0.28886 G + 0.436 B
      V = 0.615   R - 0.51499 G - 0.10001 B
    img: float tensor in [0,1], shape [B,3,H,W]
    returns: [B,3,H,W]
    """
    r = img[:, 0:1]
    g = img[:, 1:2]
    b = img[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)


def yuv_color_l2(x: torch.Tensor, y: torch.Tensor, yuv_scales: Tuple[float, float, float]) -> torch.Tensor:
    if kornia is not None:
        x_yuv = kornia.color.rgb_to_yuv(x)
        y_yuv = kornia.color.rgb_to_yuv(y)
    else:
        # Previously we compared in RGB but still used YUV scales (e.g., u=v=100),
        # which drastically over-penalized non-Y channels. Use a proper RGB->YUV fallback instead.
        x_yuv = _rgb_to_yuv_fallback(x)
        y_yuv = _rgb_to_yuv_fallback(y)
    diff = x_yuv - y_yuv
    per_channel_mse = (diff ** 2).mean(dim=[0, 2, 3])
    scales = torch.tensor(yuv_scales, dtype=per_channel_mse.dtype, device=per_channel_mse.device)
    loss = (per_channel_mse * scales).sum()
    return loss


def bce_secret_loss(logits: torch.Tensor, target_bits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target_bits)


