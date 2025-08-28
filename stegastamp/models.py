import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=True))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class StegaStampEncoder(nn.Module):
    def __init__(self, height: int, width: int, secret_size: int):
        super().__init__()
        self.height = height
        self.width = width
        self.secret_channels = 3
        # secret 稠密映射到更小的网格，然后上采样到输入分辨率
        self.secret_grid_h = max(1, height // 8)
        self.secret_grid_w = max(1, width // 8)
        secret_dense_out = self.secret_grid_h * self.secret_grid_w * self.secret_channels

        # secret -> (50, 50, 3) -> upsample x8 -> (400, 400, 3)
        self.secret_dense = nn.Sequential(
            nn.Linear(in_features=secret_size, out_features=secret_dense_out),
            nn.ReLU(inplace=True),
        )
        # 保持与TF: Dense->Reshape(grid_h,grid_w,3)->Upsample到(H,W)

        # U-Net 编码解码
        self.conv1 = conv(3 + self.secret_channels, 32)
        self.conv2 = conv(32, 32, s=2)
        self.conv3 = conv(32, 64, s=2)
        self.conv4 = conv(64, 128, s=2)
        self.conv5 = conv(128, 256, s=2)

        self.up6 = conv(256, 128, k=3, p=1)
        self.conv6 = conv(128 + 128, 128)
        self.up7 = conv(128, 64, k=3, p=1)
        self.conv7 = conv(64 + 64, 64)
        self.up8 = conv(64, 32, k=3, p=1)
        self.conv8 = conv(32 + 32, 32)
        self.up9 = conv(32, 32, k=3, p=1)
        self.conv9 = conv(32 + 32 + (3 + self.secret_channels), 32)
        self.conv10 = conv(32, 32)
        self.residual = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, secret_bits: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # secret_bits: [B, secret_size] 但为保持等价，我们输入长度=7500 的稠密向量
        # image: [B, 3, H, W] with H=W=400
        x_secret = secret_bits - 0.5
        x_image = image - 0.5

        # TF原版: Dense(7500)->(50,50,3)->Upsample x8
        x_secret = self.secret_dense(x_secret)
        b = x_secret.shape[0]
        x_secret = x_secret.view(b, self.secret_channels, self.secret_grid_h, self.secret_grid_w)
        x_secret = F.interpolate(x_secret, size=(self.height, self.width), mode="nearest")

        x_in = torch.cat([x_secret, x_image], dim=1)

        c1 = self.conv1(x_in)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        u6 = F.interpolate(c5, size=c4.shape[-2:], mode="nearest")
        u6 = self.up6(u6)
        m6 = torch.cat([c4, u6], dim=1)
        c6 = self.conv6(m6)

        u7 = F.interpolate(c6, size=c3.shape[-2:], mode="nearest")
        u7 = self.up7(u7)
        m7 = torch.cat([c3, u7], dim=1)
        c7 = self.conv7(m7)

        u8 = F.interpolate(c7, size=c2.shape[-2:], mode="nearest")
        u8 = self.up8(u8)
        m8 = torch.cat([c2, u8], dim=1)
        c8 = self.conv8(m8)

        u9 = F.interpolate(c8, size=c1.shape[-2:], mode="nearest")
        u9 = self.up9(u9)
        m9 = torch.cat([c1, u9, x_in], dim=1)
        c9 = self.conv9(m9)
        c10 = self.conv10(c9)
        residual = self.residual(c10)
        return residual


class SpatialTransformer(nn.Module):
    def __init__(self, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # theta: [B, 2, 3]
        grid = F.affine_grid(theta, size=list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size: int, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        self.freeze_stn: bool = False

        h8 = math.ceil(height / 8)
        w8 = math.ceil(width / 8)
        self.stn_params = nn.Sequential(
            conv(3, 32, s=2),
            conv(32, 64, s=2),
            conv(64, 128, s=2),
            nn.Flatten(),
            nn.Linear(128 * h8 * w8, 128),
            nn.ReLU(inplace=True),
        )
        self.fc_theta = nn.Linear(128, 6)
        nn.init.zeros_(self.fc_theta.weight)
        # 初始为单位仿射
        self.fc_theta.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.stn = SpatialTransformer(height, width)

        h32 = math.ceil(height / 32)
        w32 = math.ceil(width / 32)
        self.decoder = nn.Sequential(
            conv(3, 32, s=2),
            conv(32, 32),
            conv(32, 64, s=2),
            conv(64, 64),
            conv(64, 64, s=2),
            conv(64, 128, s=2),
            conv(128, 128, s=2),
            nn.Flatten(),
            nn.Linear(128 * h32 * w32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, secret_size),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image - 0.5
        stn_feat = self.stn_params(x)
        theta = self.fc_theta(stn_feat).view(-1, 2, 3)
        if self.freeze_stn:
            # 强制使用恒等变换（不破坏梯度到解码分支）
            identity = torch.tensor([1, 0, 0, 0, 1, 0], dtype=theta.dtype, device=theta.device).view(1, 2, 3)
            theta = identity.repeat(theta.shape[0], 1, 1)
        x_t = self.stn(x, theta)
        logits = self.decoder(x_t)
        return logits


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            conv(3, 8, s=2),
            conv(8, 16, s=2),
            conv(16, 32, s=2),
            conv(32, 64, s=2),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = image - 0.5
        fmap = self.model(x)
        out = fmap.mean()
        return out, fmap


def prepare_deployment_hiding(encoder: StegaStampEncoder, secret_dense_bits: torch.Tensor, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    residual = encoder(secret_dense_bits, image)
    encoded_image = torch.clamp(residual + image, 0.0, 1.0)
    return encoded_image, residual


def prepare_deployment_reveal(decoder: StegaStampDecoder, image: torch.Tensor) -> torch.Tensor:
    logits = decoder(image)
    return torch.round(torch.sigmoid(logits))


