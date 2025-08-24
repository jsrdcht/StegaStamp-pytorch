# 讨论 D loss 改进建议
_Exported on 2025/8/24 at GMT+8 23:37:27 from Cursor (1.4.5)_

---

**User**

你看目前代码@StegaStamp-pytorch/ 的训练 Log，可以发现 D loss 好像不怎么变化。你有什么改进的建议吗？我的代码是从@StegaStamp/ 修改过来的pytorch版本，它们使用的超参数是@base.sh 。

---

**Cursor**

我先在你的两个仓库里搜一下训练循环、判别器损失与日志输出的位置，看看 D 的计算与优化是否有问题，然后给出针对性的改进建议。随后我会并行检索相关关键字与日志格式。

我先打开你 Pytorch 版本的训练脚本与判别器定义，核对 D/G 损失、优化步数和日志打印的实现细节，然后对照原版仓库的实现。随后我会并行读取相关文件。

我已经比对了你 PyTorch 版和原 TF 版的训练实现，定位了 D 不太变化的核心原因与可操作改进点。下面按优先级给出具体建议。

### 关键差异与疑似原因
- 重要差异（判别器输入未对齐几何域）
```224:233:/workspace/StegaStamp-pytorch/stegastamp/train.py
D_real_d = discriminator(images.detach())[0]
D_fake_d = discriminator(encoded_image.detach())[0]
D_loss = D_real_d - D_fake_d
opt_d.zero_grad()
(-D_loss).backward()
for p in discriminator.parameters():
    p.data.clamp_(-0.01, 0.01)
opt_d.step()
```
```231:237:/workspace/libriries/StegaStamp/models.py
if borders == 'no_edge':
    D_output_real, _ = discriminator(image_input)
    D_output_fake, D_heatmap = discriminator(encoded_image)
else:
    D_output_real, _ = discriminator(input_warped)
    D_output_fake, D_heatmap = discriminator(encoded_warped)
```
- 你的 PyTorch 版把 real 喂入了未变换的 `images`，而 fake 喂入了已仿射的 `encoded_warped`。原版 TF 在非 `no_edge` 情况下对 real/fake 都使用了相同的几何域（`input_warped`/`encoded_warped`）。此不一致会削弱 D 的可学信号。
- 判别器训练过弱：RMSprop(lr=1e-5) + 权重裁剪 ±0.01，但没有 grad clip，也没有 n_critic，多数步 D 更新非常小，易停在近 0 区间。
- 判别器容量与损失形态保守：浅层 ReLU 小网络 + WGAN weight clipping，在 400×400、强 L2/LPIPS 约束下，D 的信号本来就弱。
- 监控不够：未区分记录 D(real)/D(fake)，只看差值容易误判“无变化”。

### 强烈建议（按优先级）
1) 让 D 看到相同几何域（最小修改且收益大）
- 用与 fake 相同的仿射域喂 real：把 `images` 改为 `apply_affine(images, theta)`（或缓存 `input_warped`），并保持 fake 用 `encoded_warped`。
- 可选：为 D 的输入也加入与 G 一致的 blur/noise/jpeg（或轻度对齐），避免 D 学到“几何/压缩差异”这种捷径。

2) 加强判别器训练强度
- 使用 Adam(lr=1e-4, betas=(0, 0.9)) 或 RMSprop(lr=5e-5)；引入 n_critic=3–5（每步先更新 D 3–5 次再更新 G 1 次）。
- 若继续使用 weight clipping，放宽到 ±0.05；更推荐去掉 weight clipping，改用正则（见下一条）。

3) 替换为更稳的对抗损失
- WGAN-GP：去掉权重裁剪，加入梯度惩罚 λ=10，优化器用 Adam(lr=1e-4, betas=(0, 0.9))，n_critic=5。
- 或 Hinge/Non-saturating + R1：D 用 hinge，G 用 -D(fake)，D 侧加 R1（γ=10）稳定训练。

4) 提升判别器表达
- 使用 LeakyReLU(0.2) 替代 ReLU；适当加深通道数（例如 8→32→64→128→256）；或给每层加 SpectralNorm（极大稳定性，替代 weight clipping）。
- 输出用 Patch 平均更合理：先对空间维求均值再对 batch 求均值（避免直接对全部维度一次性 `.mean()`）。

5) 训练权重与调度微调（配合你 base.sh）
- 保持 `--G_loss_scale 0.5` 或提升到 1.0（只影响 G，不影响 D 更新强度）。
- 保持 `--jpeg_quality 50`、`--rnd_trans 0.1`、各 ramp 与 base.sh 一致。
- batch 在 400×400 下控制在 4–8，保证判别器有稳定梯度。

6) 增强可观测性（立刻可做）
- 额外记录与打印：
  - D_real, D_fake（标成正负或直方图）
  - 判别器权重范数/梯度范数
  - G_adv 单列显示（你日志里 G 已是 `-D(fake)`，为负值属正常）
- 指标期望：调完后 D_loss 会离开 0（常见 0.01–0.2，依实现而定），D_real 与 D_fake 出现可分离，bit_acc 从 0.5 往 0.6–0.8 走再受 JPEG 等扰动回落。

### 可能的次要差异（可选修复）
- `no_im_loss_steps` 在 PyTorch 版尚未生效（原 TF 在前 1500 步只训 secret/GAN），如要完全对齐可加上这一阶段，有助于 GAN 先行收敛。
- 边缘加权 L2（`l2_edge_*`）未在 PyTorch 版实现，原版会放大边缘处的 L2，能抑制边缘伪影，也间接减轻 D 的负担。

如果你愿意，我可以按上述“最小改动方案”（统一 D 输入几何域 + n_critic + Adam + 记录 D_real/D_fake）直接帮你改训练脚本，并新增开关以无缝对齐 base.sh。要我现在就动手吗？

- 我已通读 `stegastamp/train.py` 与原库对应处，确认了 D/G 的实现及日志；接下来我可以按“最小改动方案”一次性提交对应编辑并启动一次短跑实验验证 D_loss 是否脱离 0。

```python
D_real_d = discriminator(images.detach())[0]
D_fake_d = discriminator(encoded_image.detach())[0]
D_loss = D_real_d - D_fake_d
opt_d.zero_grad()
(-D_loss).backward()
for p in discriminator.parameters():
    p.data.clamp_(-0.01, 0.01)
opt_d.step()
```

```python
if borders == 'no_edge':
    D_output_real, _ = discriminator(image_input)
    D_output_fake, D_heatmap = discriminator(encoded_image)
else:
    D_output_real, _ = discriminator(input_warped)
    D_output_fake, D_heatmap = discriminator(encoded_warped)
```

---

**User**

no_im_loss_steps 是什么意思？是不是意味着前期不训练GAN?

---

**Cursor**

不是单纯“去掉图像损失”这么简单，它也确实意味着前期不训练 GAN。

- 含义：在前 no_im_loss_steps 步中，只优化秘密恢复的交叉熵损失 secret_loss，不计算/反传图像外观损失（L2/LPIPS），也不更新判别器 D，也不让 G 的对抗项参与训练。
- 证据（原 TF 版逻辑）：
```182:188:/workspace/libriries/StegaStamp/train.py
if no_im_loss:
    _, _, global_step = sess.run([train_secret_op,loss_op,global_step_tensor],feed_dict)
else:
    _, _, global_step = sess.run([train_op,loss_op,global_step_tensor],feed_dict)
    if not args.no_gan:
        sess.run([train_dis_op, clip_D],feed_dict)
```
```135:141:/workspace/libriries/StegaStamp/train.py
train_op = tf.train.AdamOptimizer(args.lr).minimize(loss_op, var_list=g_vars, global_step=global_step_tensor)
train_secret_op = tf.train.AdamOptimizer(args.lr).minimize(secret_loss_op, var_list=g_vars, global_step=global_step_tensor)
...
train_dis_op = optimizer.apply_gradients(capped_gvs)
```
- 实际效果：前期只让编码器/解码器学会把比特解码正确，待若干步后再加入图像损失与 GAN（开始更新 D，并让 G 接收对抗梯度）。

另外，你当前的 PyTorch 版没有实现这个 warmup（虽有参数但未分支处理），因此 GAN 从第 0 步就参与训练；若要对齐原逻辑，需要加上上述“前期仅 secret_loss、禁用 D 更新与 G 的对抗项”的分支。

```python
if no_im_loss:
    _, _, global_step = sess.run([train_secret_op,loss_op,global_step_tensor],feed_dict)
else:
    _, _, global_step = sess.run([train_op,loss_op,global_step_tensor],feed_dict)
    if not args.no_gan:
        sess.run([train_dis_op, clip_D],feed_dict)
```

```python
train_op = tf.train.AdamOptimizer(args.lr).minimize(loss_op, var_list=g_vars, global_step=global_step_tensor)
train_secret_op = tf.train.AdamOptimizer(args.lr).minimize(secret_loss_op, var_list=g_vars, global_step=global_step_tensor)
...
train_dis_op = optimizer.apply_gradients(capped_gvs)
```

---

**User**

你刚刚提到的修改建议中，如下的这些我同意：
1. 用与 fake 相同的仿射域喂 real
2. 额外记录与打印 D_real, D_fake

---

**Cursor**

我将先创建两个待办项并开始实现：1) 让判别器的 real 输入使用与 fake 相同的仿射域；2) 在日志与 TensorBoard 中增加 D_real 与 D_fake 的记录与打印。接下来我会直接编辑 `stegastamp/train.py`。

我将先把判别器的 real 输入改为与 fake 相同的仿射域（使用 `input_warped`/`encoded_warped`），并在控制台与 TensorBoard 中增加 `D_real` 与 `D_fake` 的记录。随后我会检查是否有新的 linter 报错并更新待办状态。

```python
import os
import sys
import json
import time
import shutil
import glob
import argparse
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from stegastamp.models import StegaStampEncoder, StegaStampDecoder, Discriminator, prepare_deployment_hiding
from stegastamp.augment import RandomPhotometric, RandomBlurNoise, RandomJPEG, affine_matrix, apply_affine
from stegastamp.losses import LPIPSLoss, yuv_color_l2, bce_secret_loss


class ImageFolderFlat(Dataset):
    def __init__(self, root: str, size: Tuple[int, int]):
        self.files = []
        for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            ext = os.path.splitext(f)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                self.files.append(f)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        try:
            img = Image.open(fp).convert("RGB")
            img = ImageOps.fit(img, self.size)
            arr = np.array(img, dtype=np.float32) / 255.0
        except Exception:
            arr = np.zeros((self.size[1], self.size[0], 3), dtype=np.float32)
        arr = torch.from_numpy(arr).permute(2, 0, 1)
        return arr


def ramp_value(step: int, ramp: int, max_val: float) -> float:
    if ramp <= 0:
        return max_val
    return min(max_val * float(step) / float(ramp), max_val)


def main():
    parser = argparse.ArgumentParser(description="Train StegaStamp (PyTorch)")
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--train_path", type=str, default="./data/mirflickr/images1/images/")
    parser.add_argument("--logs_dir", type=str, default="./logs/")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/")
    parser.add_argument("--saved_models", type=str, default="./saved_models/")
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--run_script", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--secret_size", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=140000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2_loss_scale", type=float, default=1.5)
    parser.add_argument("--l2_loss_ramp", type=int, default=20000)
    parser.add_argument("--l2_edge_gain", type=float, default=10.0)
    parser.add_argument("--l2_edge_ramp", type=int, default=20000)
    parser.add_argument("--l2_edge_delay", type=int, default=60000)
    parser.add_argument("--lpips_loss_scale", type=float, default=1.0)
    parser.add_argument("--lpips_loss_ramp", type=int, default=20000)
    parser.add_argument("--y_scale", type=float, default=1.0)
    parser.add_argument("--u_scale", type=float, default=1.0)
    parser.add_argument("--v_scale", type=float, default=1.0)
    parser.add_argument("--secret_loss_scale", type=float, default=1.0)
    parser.add_argument("--secret_loss_ramp", type=int, default=1)
    parser.add_argument("--G_loss_scale", type=float, default=1.0)
    parser.add_argument("--G_loss_ramp", type=int, default=20000)
    parser.add_argument("--no_gan", action="store_true")
    parser.add_argument("--rnd_trans", type=float, default=0.1)
    parser.add_argument("--rnd_bri", type=float, default=0.3)
    parser.add_argument("--rnd_noise", type=float, default=0.02)
    parser.add_argument("--rnd_sat", type=float, default=1.0)
    parser.add_argument("--rnd_hue", type=float, default=0.1)
    parser.add_argument("--contrast_low", type=float, default=0.5)
    parser.add_argument("--contrast_high", type=float, default=1.5)
    parser.add_argument("--jpeg_quality", type=float, default=25)
    parser.add_argument("--no_jpeg", action="store_true")
    parser.add_argument("--rnd_trans_ramp", type=int, default=10000)
    parser.add_argument("--rnd_bri_ramp", type=int, default=1000)
    parser.add_argument("--rnd_sat_ramp", type=int, default=1000)
    parser.add_argument("--rnd_hue_ramp", type=int, default=1000)
    parser.add_argument("--rnd_noise_ramp", type=int, default=1000)
    parser.add_argument("--contrast_ramp", type=int, default=1000)
    parser.add_argument("--jpeg_quality_ramp", type=int, default=1000)
    parser.add_argument("--no_im_loss_steps", type=int, default=500)
    parser.add_argument("--pretrained", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolderFlat(args.train_path, size=(args.width, args.height))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # output directory & logs
    output_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    if args.run_script and os.path.isfile(args.run_script):
        try:
            shutil.copy2(args.run_script, os.path.join(output_dir, os.path.basename(args.run_script)))
        except Exception:
            pass
    with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    log_txt_path = os.path.join(output_dir, 'train.log')
    metrics_csv_path = os.path.join(output_dir, 'metrics.csv')
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w') as f:
            f.write('step,lr,image_loss,lpips,secret_loss,G_loss,D_loss,bit_acc,time\n')
    tb_dir = os.path.join(output_dir, 'tb')
    writer = SummaryWriter(log_dir=tb_dir)

    encoder = StegaStampEncoder(height=args.height, width=args.width, secret_size=args.secret_size).to(device)
    decoder = StegaStampDecoder(secret_size=args.secret_size, height=args.height, width=args.width).to(device)
    discriminator = Discriminator().to(device)

    lpips_loss = LPIPSLoss().to(device)
    im_photometric = RandomPhotometric(args.rnd_bri, args.rnd_sat, args.rnd_hue, args.contrast_low, args.contrast_high)
    im_blurnoise = RandomBlurNoise(max_sigma_gauss=3.0, noise_std=args.rnd_noise)
    im_jpeg = RandomJPEG(min_quality=int(args.jpeg_quality))

    g_vars = list(encoder.parameters()) + list(decoder.parameters())
    d_vars = list(discriminator.parameters())

    opt_g = optim.Adam(g_vars, lr=args.lr)
    opt_d = optim.RMSprop(d_vars, lr=1e-5)

    global_step = 0
    total_steps = max(1, len(loader))

    if args.pretrained and os.path.isfile(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"]) 
        decoder.load_state_dict(ckpt["decoder"]) 
        try:
            discriminator.load_state_dict(ckpt["discriminator"]) 
        except Exception:
            pass

    os.makedirs(os.path.join(args.checkpoints_dir, args.exp_name), exist_ok=True)

    encoder.train(); decoder.train(); discriminator.train()

    start_time = time.time()
    print(f"[init] exp={args.exp_name} images={len(dataset)} steps={args.num_steps} batch={args.batch_size}")
    with open(log_txt_path, 'a') as flog:
        flog.write(f"[init] exp={args.exp_name} images={len(dataset)} steps={args.num_steps} batch={args.batch_size}\n")

    while global_step < args.num_steps:
        for images in loader:
            if global_step >= args.num_steps:
                break
            images = images.to(device)
            b = images.size(0)

            # secrets: binary in {0,1}
            secrets = torch.randint(0, 2, (b, args.secret_size), device=device).float()

            l2_loss_scale = ramp_value(global_step, args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = ramp_value(global_step, args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = ramp_value(global_step, args.secret_loss_ramp, args.secret_loss_scale)
            G_loss_scale = ramp_value(global_step, args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = args.l2_edge_gain if global_step > args.l2_edge_delay else 0.0

            rnd_tran = ramp_value(global_step, args.rnd_trans_ramp, args.rnd_trans)
            max_translate = (args.width * rnd_tran)
            theta = affine_matrix(b, max_translate, args.height, args.width, device)

            # encode
            input_warped = apply_affine(images, theta)
            residual_warped = encoder(secrets, input_warped)
            encoded_warped = torch.clamp(residual_warped + input_warped, 0.0, 1.0)
            residual = apply_affine(residual_warped, torch.inverse(torch.cat([theta, torch.tensor([0,0,1],device=device).view(1,1,3).repeat(b,1,1)], dim=1))[:, :2, :]) if False else residual_warped
            # no_edge / black 边界模式：对齐 TF 实现我们采用 black（更贴近ISSBA脚本）
            encoded_image = encoded_warped

            # transformer
            transformed = im_blurnoise(encoded_image, scale=rnd_tran)
            transformed = im_photometric(transformed, scale=1.0)
            if not args.no_jpeg:
                transformed = im_jpeg(transformed, scale=1.0)

            # decode
            decoded_logits = decoder(transformed)

            # losses
            image_loss = yuv_color_l2(encoded_image, images, yuv_scales=(args.y_scale, args.u_scale, args.v_scale))
            lpips_val = lpips_loss(encoded_image * 2 - 1, images * 2 - 1)  # lpips expects [-1,1]
            secret_loss = bce_secret_loss(decoded_logits, secrets)

            # Generator step (freeze D)
            for p in discriminator.parameters():
                p.requires_grad = False
            D_fake_g = discriminator(encoded_image)[0]
            # WGAN generator wants to maximize D(fake) => minimize -D(fake)
            G_adv = -D_fake_g
            loss_g = l2_loss_scale * image_loss + lpips_loss_scale * lpips_val + secret_loss_scale * secret_loss
            if not args.no_gan:
                loss_g = loss_g + G_loss_scale * G_adv
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            for p in discriminator.parameters():
                p.requires_grad = True

            # Discriminator step (detach G output)
            D_real_d = discriminator(input_warped.detach())[0]
            D_fake_d = discriminator(encoded_image.detach())[0]
            D_loss = D_real_d - D_fake_d
            opt_d.zero_grad()
            (-D_loss).backward()
            # weight clip like TF code
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
            opt_d.step()

            # logging
            if (global_step % args.log_interval) == 0:
                elapsed = time.time() - start_time
                with torch.no_grad():
                    bit_acc = float((torch.sigmoid(decoded_logits).round() == secrets).float().mean().item())
                msg = (f"step {global_step} | lr {opt_g.param_groups[0]['lr']:.2e} | "
                       f"im {image_loss.item():.4f} lpips {lpips_val.item():.4f} sec {secret_loss.item():.4f} "
                       f"G {G_adv.item():.4f} D {D_loss.item():.4f} Dreal {D_real_d.item():.4f} Dfake {D_fake_d.item():.4f} "
                       f"bit {bit_acc:.4f} | t {elapsed:.1f}s")
                print(msg)
                with open(log_txt_path, 'a') as flog:
                    flog.write(msg + "\n")
                with open(metrics_csv_path, 'a') as fcsv:
                    fcsv.write(f"{global_step},{opt_g.param_groups[0]['lr']:.6f},{image_loss.item():.6f},{lpips_val.item():.6f},{secret_loss.item():.6f},{G_adv.item():.6f},{D_loss.item():.6f},{bit_acc:.6f},{elapsed:.2f}\n")
                # tensorboard scalars
                writer.add_scalar('loss/image', image_loss.item(), global_step)
                writer.add_scalar('loss/lpips', lpips_val.item(), global_step)
                writer.add_scalar('loss/secret', secret_loss.item(), global_step)
                writer.add_scalar('loss/G', G_adv.item(), global_step)
                writer.add_scalar('loss/D', D_loss.item(), global_step)
                writer.add_scalar('loss/D_real', D_real_d.item(), global_step)
                writer.add_scalar('loss/D_fake', D_fake_d.item(), global_step)
                writer.add_scalar('metrics/bit_acc', bit_acc, global_step)
                writer.add_scalar('opt/lr', opt_g.param_groups[0]['lr'], global_step)
                # tensorboard images（下采样避免占用过多存储）
                try:
                    grid_in = torch.clamp(images[:4], 0, 1)
                    grid_en = torch.clamp(encoded_image[:4], 0, 1)
                    writer.add_images('images/input', grid_in, global_step)
                    writer.add_images('images/encoded', grid_en, global_step)
                except Exception:
                    pass

            if (global_step % args.save_interval) == 0:
                ckpt = {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "step": global_step,
                }
                torch.save(ckpt, os.path.join(args.checkpoints_dir, args.exp_name, f"{args.exp_name}_{global_step}.pt"))

            global_step += 1

    # save deploy models
    os.makedirs(os.path.join(args.saved_models, args.exp_name), exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, os.path.join(args.saved_models, args.exp_name, "stegastamp.pt"))
    try:
        writer.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()



```

我刚完成代码编辑，接下来我会把“统一判别器 real 输入到与 fake 相同的仿射域”与“增加 D_real / D_fake 的日志打印与 TensorBoard 标量”标记为完成，然后给你简要说明如何验证改动效果。

- 我已将 D 的 real 输入改为与 fake 相同的仿射域：`real = discriminator(input_warped)`，`fake = discriminator(encoded_warped)`。同时按现有流程在 G 分支里继续使用 `encoded_image` 做解码和损失，确保功能不变。
- 我增加了日志与 TensorBoard 记录：新增打印 `Dreal`、`Dfake`，并写入 `loss/D_real` 与 `loss/D_fake` 标量。

验证建议：
- 重新开一轮训练，观察日志中 `Dreal` 与 `Dfake` 是否出现可分离（不再长时间几乎相等），`D_loss = Dreal - Dfake` 不再近 0。
- 用 TensorBoard 打开 `output/<exp>/tb`，查看 `loss/D_real`、`loss/D_fake` 曲线。

如果你需要，我可以顺手把 `no_im_loss_steps` 的 warmup 分支加上，这样前 1500 步只训 secret，不更新 D，也不将对抗项加到 G。

