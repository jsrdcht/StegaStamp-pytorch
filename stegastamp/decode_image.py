import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import torch
import bchlib

from .models import StegaStampDecoder


BCH_POLYNOMIAL = 137
BCH_BITS = 5


def bits_to_bytes(bits):
    out = bytearray()
    for b in range(0, len(bits), 8):
        byte = 0
        for i in range(8):
            if b + i < len(bits):
                byte = (byte << 1) | bits[b + i]
        out.append(byte)
    return out


def main():
    parser = argparse.ArgumentParser(description="Decode image with PyTorch StegaStamp")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--secret_size", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = StegaStampDecoder(secret_size=args.secret_size, height=args.height, width=args.width).to(device)

    ckpt = torch.load(args.model, map_location="cpu")
    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"]) 
    else:
        decoder.load_state_dict(ckpt)
    decoder.eval()

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    size = (args.width, args.height)
    with torch.no_grad():
        image = Image.open(args.image).convert("RGB")
        image = ImageOps.fit(image, size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device)

        logits = decoder(image)
        bits = torch.round(torch.sigmoid(logits)).squeeze(0).cpu().numpy().astype(np.uint8).tolist()

        # 先取前 56 + ecc 位尝试纠错
        raw = bits_to_bytes(bits)
        data, ecc = raw[:7], raw[7:]
        try:
            bch.decode(bytearray(data), bytearray(ecc))
            msg = bytes(data).decode('utf-8').strip()
            print(msg)
        except Exception:
            print("decode failed")


if __name__ == "__main__":
    main()


