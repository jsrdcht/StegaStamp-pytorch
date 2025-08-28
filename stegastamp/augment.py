from typing import Tuple
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

# 不在顶层导入第三方图像库，避免静态分析无法解析依赖
DiffJPEG = None


class RandomPhotometric(nn.Module):
    def __init__(self,
                 rnd_bri: float = 0.3,
                 rnd_sat: float = 1.0,
                 rnd_hue: float = 0.1,
                 contrast_low: float = 0.5,
                 contrast_high: float = 1.5):
        super().__init__()
        self.rnd_bri = rnd_bri
        self.rnd_sat = rnd_sat
        self.rnd_hue = rnd_hue
        self.contrast_low = contrast_low
        self.contrast_high = contrast_high

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        b = x.shape[0]
        # brightness
        bri = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 2 * (self.rnd_bri * scale)
        x = torch.clamp(x + bri, 0.0, 1.0)
        # contrast
        cmin = 1.0 - (1.0 - self.contrast_low) * scale
        cmax = 1.0 + (self.contrast_high - 1.0) * scale
        c = torch.empty(b, 1, 1, 1, device=x.device).uniform_(cmin, cmax)
        x = torch.clamp(x * c, 0.0, 1.0)
        # saturation（近似）：围绕亮度放大色度分量
        lum = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        s_scale = torch.rand(b, 1, 1, 1, device=x.device) * (self.rnd_sat * scale)
        x = torch.clamp((x - lum) * (1.0 + s_scale) + lum, 0.0, 1.0)
        # hue: 在 YUV 色度平面上做小角度旋转，近似模拟色相变化
        if self.rnd_hue > 0 and scale > 0:
            # 角度范围 [-rnd_hue*pi, rnd_hue*pi]
            delta = (torch.rand(b, 1, 1, 1, device=x.device) * 2.0 - 1.0) * (self.rnd_hue * scale) * math.pi
            cos_a = torch.cos(delta)
            sin_a = torch.sin(delta)
            r = x[:, 0:1]
            g = x[:, 1:2]
            bch = x[:, 2:3]
            # RGB -> YUV
            y = lum
            u = -0.14713 * r - 0.28886 * g + 0.436 * bch
            v =  0.615   * r - 0.51499 * g - 0.10001 * bch
            u2 = u * cos_a - v * sin_a
            v2 = u * sin_a + v * cos_a
            # YUV -> RGB（近似逆变换）
            r2 = y + 1.13983 * v2
            g2 = y - 0.39465 * u2 - 0.58060 * v2
            b2 = y + 2.03211 * u2
            x = torch.clamp(torch.cat([r2, g2, b2], dim=1), 0.0, 1.0)
        return x


class RandomBlurNoise(nn.Module):
    def __init__(self, max_sigma_gauss: float = 3.0, noise_std: float = 0.02):
        super().__init__()
        self.max_sigma_gauss = max_sigma_gauss
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        # 当 scale<=0 时，完全跳过模糊与噪声，避免在 warmup 阶段抹除编码痕迹
        if self.max_sigma_gauss > 0 and scale > 0:
            sigma = (torch.rand(1, device=x.device) * (self.max_sigma_gauss * scale)).item()
            if sigma > 0:
                ksize = int(2 * math.ceil(3 * float(sigma)) + 1)
                ksize = max(3, min(ksize, 31))
                # 构造高斯核
                ax = torch.arange(ksize, device=x.device) - (ksize - 1) / 2.0
                gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
                kernel2d = (gauss[:, None] @ gauss[None, :])
                kernel2d = kernel2d / kernel2d.sum()
                kernel = kernel2d.view(1, 1, ksize, ksize)
                kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # groups=channels
                x = F.conv2d(x, kernel, padding=ksize // 2, groups=x.shape[1])
        if self.noise_std > 0 and scale > 0:
            std = torch.rand(1, device=x.device) * (self.noise_std * scale)
            noise = torch.randn_like(x) * std
            x = torch.clamp(x + noise, 0.0, 1.0)
        return x


class RandomJPEG(nn.Module):
    def __init__(self, min_quality: int = 25):
        super().__init__()
        self.min_quality = min_quality
        global DiffJPEG
        # 延迟导入，避免静态分析报错
        if DiffJPEG is None:
            try:
                import importlib
                DiffJPEG = importlib.import_module('diffjpeg').DiffJPEG
            except Exception:
                DiffJPEG = None
        self.jpeg = DiffJPEG(differentiable=True) if DiffJPEG is not None else None

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        if self.jpeg is None:
            return x
        # quality in [min_quality, 100]
        q = 100.0 - torch.rand(1, device=x.device) * (100.0 - float(self.min_quality)) * scale
        q = torch.clamp(q, float(self.min_quality), 100.0)
        # DiffJPEG expects [0,1]
        return torch.clamp(self.jpeg(x, quality=q.item()), 0.0, 1.0)


def affine_matrix(batch: int, max_translate_px: float, height: int, width: int, device) -> torch.Tensor:
    # random small affine: translation only (scale & rotation omitted for simplicity)
    tx = (torch.rand(batch, device=device) - 0.5) * 2 * (max_translate_px / (width / 2))
    ty = (torch.rand(batch, device=device) - 0.5) * 2 * (max_translate_px / (height / 2))
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    return theta


def apply_affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, size=list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_affine_mask_like(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    生成与 x 同形状的有效区域掩码：1 表示有效采样区域，0 表示越界。
    注意掩码必须用 zeros padding 来区分越界区域。
    """
    ones = torch.ones_like(x)
    grid = F.affine_grid(theta, size=list(x.shape), align_corners=False)
    mask = F.grid_sample(ones, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # 少量数值抖动，阈值化到 [0,1]
    return (mask > 0.999).float()


def invert_affine_2x3(theta: torch.Tensor) -> torch.Tensor:
    # theta: [N, 2, 3] mapping output coords -> input coords in normalized space
    A = theta[:, :, :2]               # [N,2,2]
    t = theta[:, :, 2:]               # [N,2,1]
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    invA = torch.zeros_like(A)
    invA[:, 0, 0] =  A[:, 1, 1] / det
    invA[:, 1, 1] =  A[:, 0, 0] / det
    invA[:, 0, 1] = -A[:, 0, 1] / det
    invA[:, 1, 0] = -A[:, 1, 0] / det
    invt = -torch.bmm(invA, t).squeeze(-1)   # [N,2]
    theta_inv = torch.cat([invA, invt.unsqueeze(-1)], dim=2)
    return theta_inv

