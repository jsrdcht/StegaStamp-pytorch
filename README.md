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
- Removed the original TensorFlow pipeline's affine and inverse transforms around the encoder/decoder. This weakens geometric alignment cues and can make optimization harder (slower convergence, more sensitive to learning rate/regularization).

### Notes
- For ISSBA: feed the generated `*_hidden.png` into ISSBA's classifier backdoor training script (same workflow as the original repo).


