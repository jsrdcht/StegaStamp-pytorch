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
from stegastamp.augment import RandomPhotometric, RandomBlurNoise, RandomJPEG, affine_matrix, apply_affine, invert_affine_2x3, apply_affine_mask_like
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
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--logs_dir", type=str, default="./logs/")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/")
    parser.add_argument("--saved_models", type=str, default="./saved_models/")
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--run_script", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--val_interval", type=int, default=10000)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--secret_size", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=140000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2_loss_scale", type=float, default=2.0)
    parser.add_argument("--l2_loss_ramp", type=int, default=15000)
    parser.add_argument("--l2_edge_gain", type=float, default=10.0)
    parser.add_argument("--l2_edge_ramp", type=int, default=10000)
    parser.add_argument("--l2_edge_delay", type=int, default=80000)
    parser.add_argument("--lpips_loss_scale", type=float, default=1.5)
    parser.add_argument("--lpips_loss_ramp", type=int, default=15000)
    parser.add_argument("--y_scale", type=float, default=1.0)
    parser.add_argument("--u_scale", type=float, default=100.0)
    parser.add_argument("--v_scale", type=float, default=100.0)
    parser.add_argument("--secret_loss_scale", type=float, default=1.5)
    parser.add_argument("--secret_loss_ramp", type=int, default=1)
    parser.add_argument("--G_loss_scale", type=float, default=0.5)
    parser.add_argument("--G_loss_ramp", type=int, default=15000)
    parser.add_argument("--no_gan", action="store_true")
    parser.add_argument("--rnd_trans", type=float, default=0.1)
    parser.add_argument("--rnd_bri", type=float, default=0.3)
    parser.add_argument("--rnd_noise", type=float, default=0.02)
    parser.add_argument("--rnd_sat", type=float, default=1.0)
    parser.add_argument("--rnd_hue", type=float, default=0.1)
    parser.add_argument("--contrast_low", type=float, default=0.5)
    parser.add_argument("--contrast_high", type=float, default=1.5)
    parser.add_argument("--jpeg_quality", type=float, default=50)
    parser.add_argument("--no_jpeg", action="store_true")
    parser.add_argument("--rnd_trans_ramp", type=int, default=10000)
    parser.add_argument("--rnd_bri_ramp", type=int, default=1000)
    parser.add_argument("--rnd_sat_ramp", type=int, default=1000)
    parser.add_argument("--rnd_hue_ramp", type=int, default=1000)
    parser.add_argument("--rnd_noise_ramp", type=int, default=1000)
    parser.add_argument("--contrast_ramp", type=int, default=1000)
    parser.add_argument("--jpeg_quality_ramp", type=int, default=1000)
    parser.add_argument("--no_im_loss_steps", type=int, default=20000)
    parser.add_argument("--pretrained", type=str, default=None)
    # dataset sampling
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=1337)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.exp_name = f"{timestamp}_{args.exp_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolderFlat(args.train_path, size=(args.width, args.height))
    orig_images = len(dataset)
    # training-time dataset subsampling
    if (args.sample_count is not None and args.sample_count > 0) or (args.sample_ratio is not None and args.sample_ratio < 1.0):
        try:
            target_k = args.sample_count if (args.sample_count is not None and args.sample_count > 0) else int(max(1, orig_images * float(args.sample_ratio)))
            target_k = max(1, min(target_k, orig_images))
            rng = random.Random(int(args.sample_seed) if args.sample_seed is not None else None)
            dataset.files = rng.sample(dataset.files, target_k)
        except Exception:
            pass
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # validation loader
    val_loader = None
    if args.val_path is not None and os.path.isdir(args.val_path):
        try:
            val_dataset = ImageFolderFlat(args.val_path, size=(args.width, args.height))
            if len(val_dataset) > 0:
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)
        except Exception:
            val_loader = None

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
    # 恢复摄影增广模块
    im_photometric = RandomPhotometric(args.rnd_bri, args.rnd_sat, args.rnd_hue, args.contrast_low, args.contrast_high)
    im_blurnoise = RandomBlurNoise(max_sigma_gauss=3.0, noise_std=args.rnd_noise)
    im_jpeg = RandomJPEG(min_quality=int(args.jpeg_quality))

    g_vars = list(encoder.parameters()) + list(decoder.parameters())
    d_vars = list(discriminator.parameters())

    opt_g = optim.Adam(g_vars, lr=args.lr)
    opt_d = optim.RMSprop(d_vars, lr=1e-5)
    
    # Warmup期间专门用于训练secret恢复的优化器
    opt_g_secret = optim.Adam(g_vars, lr=args.lr)
    
    # 对抗训练平衡计数器
    d_loss_history = []

    # 最近一次反传的梯度范数（用于日志核查是否在训练）
    last_grad_enc = 0.0
    last_grad_dec = 0.0

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

    # 移除边缘掩码与边缘L2

    start_time = time.time()
    sample_info = ""
    if len(dataset) != orig_images:
        sample_val = args.sample_count if (args.sample_count is not None and args.sample_count > 0) else args.sample_ratio
        sample_info = f" sample={sample_val} seed={args.sample_seed} (orig {orig_images})"
    print(f"[init] exp={args.exp_name} images={len(dataset)} steps={args.num_steps} batch={args.batch_size}{sample_info}")
    with open(log_txt_path, 'a') as flog:
        flog.write(f"[init] exp={args.exp_name} images={len(dataset)} steps={args.num_steps} batch={args.batch_size}{sample_info}\n")

    # best validation tracker
    best_val_lpips = float('inf')
    best_val_step = -1
    best_val_bit_acc = 0.0

    def keep_only_latest_and_best(directory: str) -> None:
        """Remove all checkpoint files except 'latest.pth' and 'best.pth'."""
        try:
            for fname in os.listdir(directory):
                if fname.endswith((".pt", ".pth")) and (fname not in ("best.pth", "latest.pth")):
                    fpath = os.path.join(directory, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
        except Exception:
            pass

    def evaluate_on_val() -> Tuple[float, float, float, float]:
        if val_loader is None:
            return float('inf'), 0.0, float('inf'), float('inf')
        enc_was_train = encoder.training
        dec_was_train = decoder.training
        encoder.eval(); decoder.eval()
        val_lpips_sum = 0.0
        val_img_loss_sum = 0.0
        val_secret_loss_sum = 0.0
        total_bits = 0
        correct_bits = 0
        seen_imgs = 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                b = images.size(0)
                secrets = torch.randint(0, 2, (b, args.secret_size), device=device).float()
                residual = encoder(secrets, images)
                encoded_image = torch.clamp(residual + images, 0.0, 1.0)
                decoded_logits = decoder(encoded_image)
                img_loss = yuv_color_l2(encoded_image, images, yuv_scales=(args.y_scale, args.u_scale, args.v_scale))
                lp = lpips_loss(encoded_image * 2 - 1, images * 2 - 1)
                sec_loss = bce_secret_loss(decoded_logits, secrets)
                preds = torch.sigmoid(decoded_logits).round()
                correct_bits += int((preds == secrets).float().sum().item())
                total_bits += int(secrets.numel())
                val_img_loss_sum += float(img_loss.item()) * b
                val_lpips_sum += float(lp.item()) * b
                val_secret_loss_sum += float(sec_loss.item()) * b
                seen_imgs += int(b)
        if enc_was_train:
            encoder.train()
        if dec_was_train:
            decoder.train()
        mean_img_loss = val_img_loss_sum / max(1, seen_imgs)
        mean_lpips = val_lpips_sum / max(1, seen_imgs)
        mean_secret_loss = val_secret_loss_sum / max(1, seen_imgs)
        bit_acc = (correct_bits / max(1, total_bits))
        return mean_lpips, bit_acc, mean_secret_loss, mean_img_loss

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

            no_im_loss = global_step < args.no_im_loss_steps

            # 去掉几何：统一直接在原图域编码；保留掩码合成结构（恒等掩码）
            input_warped = images
            residual_warped = encoder(secrets, input_warped)
            encoded_warped = residual_warped + input_warped
            mask_back = torch.ones_like(images[:, :1, :, :])
            encoded_image = encoded_warped * mask_back + images * (1.0 - mask_back)

            # 送入解码器与判别器前进行 clamp，避免越界值导致训练不稳定
            encoded_warped = torch.clamp(encoded_warped, 0.0, 1.0)
            encoded_image = torch.clamp(encoded_image, 0.0, 1.0)

            # transformer：仅保留摄影增广
            noise_scale = ramp_value(global_step, args.rnd_noise_ramp, 1.0)
            photo_scale = min(
                ramp_value(global_step, args.rnd_bri_ramp, 1.0),
                ramp_value(global_step, args.rnd_sat_ramp, 1.0),
                ramp_value(global_step, args.rnd_hue_ramp, 1.0),
                ramp_value(global_step, args.contrast_ramp, 1.0),
            )
            jpeg_scale = ramp_value(global_step, args.jpeg_quality_ramp, 1.0)

            transformed = encoded_image
            if not no_im_loss:
                transformed = im_blurnoise(transformed, scale=noise_scale)
                transformed = im_photometric(transformed, scale=photo_scale)
                if not args.no_jpeg:
                    transformed = im_jpeg(transformed, scale=jpeg_scale)

            # decode
            decoded_logits = decoder(transformed)

            # losses（不使用YUV图像差异，改为 residual L2）
            residual = residual_warped
            l2_loss = (residual ** 2).mean()
            lpips_val = lpips_loss(encoded_image * 2 - 1, images * 2 - 1)
            secret_loss = bce_secret_loss(decoded_logits, secrets)

            # 不使用边缘L2

            no_im_loss = global_step < args.no_im_loss_steps
            # Generator step (freeze D)
            for p in discriminator.parameters():
                p.requires_grad = False
            D_fake_g = discriminator(encoded_warped)[0]
            # WGAN generator wants to maximize D(fake) => minimize -D(fake)
            G_adv = -D_fake_g
            
            if no_im_loss:
                # warmup: 仅训练 secret 恢复，使用专门的优化器
                decoder.freeze_stn = True
                opt_g_secret.zero_grad()
                secret_loss.backward()
                # 记录梯度范数
                with torch.no_grad():
                    ge_sum = 0.0
                    for p in encoder.parameters():
                        if p.grad is not None:
                            ge_sum += float(p.grad.detach().norm(2).item() ** 2)
                    gd_sum = 0.0
                    for p in decoder.parameters():
                        if p.grad is not None:
                            gd_sum += float(p.grad.detach().norm(2).item() ** 2)
                    last_grad_enc = ge_sum ** 0.5
                    last_grad_dec = gd_sum ** 0.5
                opt_g_secret.step()
            else:
                loss_g = l2_loss_scale * l2_loss + lpips_loss_scale * lpips_val + secret_loss_scale * secret_loss
                if not args.no_gan:
                    loss_g = loss_g + G_loss_scale * G_adv
                decoder.freeze_stn = False
                opt_g.zero_grad()
                loss_g.backward()
                # 记录梯度范数
                with torch.no_grad():
                    ge_sum = 0.0
                    for p in encoder.parameters():
                        if p.grad is not None:
                            ge_sum += float(p.grad.detach().norm(2).item() ** 2)
                    gd_sum = 0.0
                    for p in decoder.parameters():
                        if p.grad is not None:
                            gd_sum += float(p.grad.detach().norm(2).item() ** 2)
                    last_grad_enc = ge_sum ** 0.5
                    last_grad_dec = gd_sum ** 0.5
                opt_g.step()
            for p in discriminator.parameters():
                p.requires_grad = True

            # Discriminator step（取消是否训练D的门控）：在非warmup且启用GAN时每步训练D
            if not no_im_loss and not args.no_gan:
                D_real_d = discriminator(input_warped.detach())[0]
                D_fake_d = discriminator(encoded_warped.detach())[0]
                D_loss = D_real_d - D_fake_d
                opt_d.zero_grad()
                (-D_loss).backward()
                torch.nn.utils.clip_grad_value_(discriminator.parameters(), 0.25)
                opt_d.step()
                with torch.no_grad():
                    for p in discriminator.parameters():
                        p.clamp_(-0.01, 0.01)
            else:
                # warmup或关闭GAN时，仅统计
                with torch.no_grad():
                    D_real_d = discriminator(input_warped.detach())[0]
                    D_fake_d = discriminator(encoded_warped.detach())[0]
                    D_loss = D_real_d - D_fake_d

            # logging
            if (global_step % args.log_interval) == 0:
                elapsed = time.time() - start_time
                with torch.no_grad():
                    bit_acc = float((torch.sigmoid(decoded_logits).round() == secrets).float().mean().item())
                    if no_im_loss:
                        total_loss = secret_loss.item()
                    else:
                        total_expr = (l2_loss_scale * l2_loss + lpips_loss_scale * lpips_val + secret_loss_scale * secret_loss)
                        if not args.no_gan:
                            total_expr = total_expr + G_loss_scale * G_adv
                        total_loss = total_expr.item()
                msg = (f"step {global_step} | lr {opt_g.param_groups[0]['lr']:.2e} | "
                       f"im {l2_loss.item():.4f} lpips {lpips_val.item():.4f} sec {secret_loss.item():.4f} "
                       f"G {G_adv.item():.4f} D {D_loss.item():.4f} Dreal {D_real_d.item():.4f} Dfake {D_fake_d.item():.4f} "
                       f"total {total_loss:.4f} bit {bit_acc:.4f} | "
                       f"grad enc {last_grad_enc:.3e} dec {last_grad_dec:.3e} | t {elapsed:.1f}s")
                print(msg)
                with open(log_txt_path, 'a') as flog:
                    flog.write(msg + "\n")
                with open(metrics_csv_path, 'a') as fcsv:
                    fcsv.write(f"{global_step},{opt_g.param_groups[0]['lr']:.6f},{l2_loss.item():.6f},{lpips_val.item():.6f},{secret_loss.item():.6f},{G_adv.item():.6f},{D_loss.item():.6f},{bit_acc:.6f},{elapsed:.2f}\n")
                # tensorboard scalars
                writer.add_scalar('loss/total', total_loss, global_step)
                writer.add_scalar('loss/image', l2_loss.item(), global_step)
                writer.add_scalar('loss/lpips', lpips_val.item(), global_step)
                writer.add_scalar('loss/secret', secret_loss.item(), global_step)
                writer.add_scalar('loss/G', G_adv.item(), global_step)
                writer.add_scalar('loss/D', D_loss.item(), global_step)
                writer.add_scalar('loss/D_real', D_real_d.item(), global_step)
                writer.add_scalar('loss/D_fake', D_fake_d.item(), global_step)
                writer.add_scalar('metrics/bit_acc', bit_acc, global_step)
                # 权重监控
                writer.add_scalar('weights/l2', l2_loss_scale, global_step)
                writer.add_scalar('weights/lpips', lpips_loss_scale, global_step)
                writer.add_scalar('weights/G', G_loss_scale, global_step)
                writer.add_scalar('opt/lr', opt_g.param_groups[0]['lr'], global_step)
                writer.add_scalar('training/warmup', 1.0 if no_im_loss else 0.0, global_step)
                writer.add_scalar('grad/encoder_l2', last_grad_enc, global_step)
                writer.add_scalar('grad/decoder_l2', last_grad_dec, global_step)
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
                latest_path = os.path.join(args.checkpoints_dir, args.exp_name, "latest.pth")
                torch.save(ckpt, latest_path)
                keep_only_latest_and_best(os.path.join(args.checkpoints_dir, args.exp_name))

            # validation and best checkpointing
            if val_loader is not None and (global_step % args.val_interval) == 0:
                v_lpips, v_bit_acc, v_sec_loss, v_im_loss = evaluate_on_val()
                # log to console and files
                v_msg = (f"[val] step {global_step} | lpips {v_lpips:.4f} image {v_im_loss:.4f} secret {v_sec_loss:.4f} bit {v_bit_acc:.4f}")
                print(v_msg)
                with open(log_txt_path, 'a') as flog:
                    flog.write(v_msg + "\n")
                writer.add_scalar('val/lpips', v_lpips, global_step)
                writer.add_scalar('val/image', v_im_loss, global_step)
                writer.add_scalar('val/secret', v_sec_loss, global_step)
                writer.add_scalar('val/bit_acc', v_bit_acc, global_step)
                # selection: only consider checkpoints with bit acc >= 0.99 and lpips minimal
                if (v_bit_acc >= 0.99) and (v_lpips < best_val_lpips):
                    best_val_lpips = v_lpips
                    best_val_step = global_step
                    best_val_bit_acc = v_bit_acc
                    best_ckpt = {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "step": global_step,
                        "val_lpips": v_lpips,
                        "val_bit_acc": v_bit_acc,
                        "val_image_loss": v_im_loss,
                        "val_secret_loss": v_sec_loss,
                    }
                    torch.save(best_ckpt, os.path.join(args.checkpoints_dir, args.exp_name, "best.pth"))
                    keep_only_latest_and_best(os.path.join(args.checkpoints_dir, args.exp_name))
                    print(f"[val] New best.pth at step {global_step}: lpips {v_lpips:.4f}, bit {v_bit_acc:.4f}")

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




