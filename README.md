## StegaStamp-pytorch

PyTorch 复现 StegaStamp（对齐 ISSBA 使用的方案），提供训练与推理脚本。

### 项目状态
- 当前处于开发阶段（WIP），接口与行为可能变更；欢迎提交 PR。
- 已知问题：训练中 secret loss 下降缓慢/困难，欢迎一起讨论与改进。

### 环境（pixi）

1) 安装 pixi 后，在项目根目录执行：

```bash
pixi install
```

2) 运行训练/推理帮助：

```bash
pixi run train
pixi run encode
pixi run decode
```

### 训练

```bash
python -m stegastamp.train exp_name \
  --train_path ./data/mirflickr/images1/images/ \
  --height 400 --width 400 \
  --secret_size 20 \
  --num_steps 140000 --batch_size 4 --lr 1e-4 \
  --no_gan  # 如需GAN，删除该开关
```

输出：`checkpoints/exp_name/*.pt` 与 `saved_models/exp_name/stegastamp.pt`

### 推理（生成隐藏图与残差）

```bash
python -m stegastamp.encode_image --model saved_models/exp_name/stegastamp.pt \
  --images_dir ./examples \
  --save_dir ./outputs \
  --secret Stega!! \
  --height 400 --width 400
```

### 解码

```bash
python -m stegastamp.decode_image --model saved_models/exp_name/stegastamp.pt \
  --image ./outputs/xxx_hidden.png \
  --height 400 --width 400 --secret_size 20
```

### 与原TF实现的差异
- 训练图像扰动使用 kornia/diffjpeg 近似 tf.contrib 与自定义 JPEG/模糊。
- LPIPS 使用 `lpips` 包；若未安装会退化到 L2。
- 解码器含 STN（仿射校正）并输出 `secret_size` 维度 logits；加 `sigmoid` 后取整。
- 编码器内部 secret 稠密向量长度固定 7500（与 TF Dense 一致），ECC 打包后位数用 0 填充到 7500。

### 说明
- 若用于 ISSBA：请将本项目生成的 `*_hidden.png` 喂给 ISSBA 的分类器后门训练脚本（与原仓库流程一致）。


