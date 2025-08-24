## StegaStamp-pytorch

A PyTorch reimplementation of StegaStamp aligned with the ISSBA workflow. It provides training and inference scripts.

### Project status
- Work in progress (WIP). APIs and behavior may change. PRs are welcome.
- Known issue: the secret loss tends to decrease very slowly and is hard to optimize. Contributions and discussions are welcome.

### Environment (pixi)

1) After installing pixi, run at the project root:

```bash
pixi install
```

2) Helpful entrypoints:

```bash
pixi run train
pixi run encode
pixi run decode
```

### Training

```bash
bash scripts/train.sh
```

Outputs: `checkpoints/exp_name/*.pt` and `saved_models/exp_name/stegastamp.pt`

### Inference (generate hidden image and residual)

```bash
python -m stegastamp.encode_image --model saved_models/exp_name/stegastamp.pt \
  --images_dir ./examples \
  --save_dir ./outputs \
  --secret Stega!! \
  --height 400 --width 400
```

### Decode

```bash
python -m stegastamp.decode_image --model saved_models/exp_name/stegastamp.pt \
  --image ./outputs/xxx_hidden.png \
  --height 400 --width 400 --secret_size 20
```

### Differences from the original TF implementation
- Training perturbations use kornia/diffjpeg to approximate tf.contrib and custom JPEG/blur ops.
- LPIPS via the `lpips` package; if not installed it falls back to L2.
- The decoder contains an STN (affine correction) and outputs `secret_size`-dim logits; apply sigmoid and threshold to bits.
- The encoder uses a fixed dense secret vector length of 7500 (matching TF Dense); ECC-packed bits are zero-padded to 7500.

### Notes
- For ISSBA: feed the generated `*_hidden.png` into ISSBA's classifier backdoor training script (same workflow as the original repo).


