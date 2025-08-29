import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import torch
import bchlib

from .models import StegaStampEncoder, prepare_deployment_hiding


BCH_POLYNOMIAL = 137
BCH_BITS = 5


def main():
    parser = argparse.ArgumentParser(description="Encode image with PyTorch StegaStamp")
    parser.add_argument("--model", type=str, required=True, help="path to saved stegastamp.pt or encoder weights")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--secret", type=str, default="Stega!!")
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--secret_size", type=int, default=100)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = [os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir)]
    else:
        raise SystemExit("Missing input image(s)")

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = StegaStampEncoder(height=args.height, width=args.width, secret_size=args.secret_size).to(device)

    ckpt = torch.load(args.model, map_location="cpu")
    if "encoder" in ckpt:
        encoder.load_state_dict(ckpt["encoder"]) 
    else:
        encoder.load_state_dict(ckpt)

    encoder.eval()

    bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
    if len(args.secret) > 7:
        raise SystemExit("Error: max 7 characters with ECC")

    data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0,0,0,0])

    secret = torch.tensor(secret, dtype=torch.float32, device=device).unsqueeze(0)
    # pad/trim to secret_size
    if secret.shape[1] < args.secret_size:
        pad = torch.zeros(1, args.secret_size - secret.shape[1], device=device)
        secret = torch.cat([secret, pad], dim=1)
    else:
        secret = secret[:, :args.secret_size]

    size = (args.width, args.height)
    with torch.no_grad():
        for filename in files_list:
            image = Image.open(filename).convert("RGB")
            image = ImageOps.fit(image, size)
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device)

            hidden_img, residual = prepare_deployment_hiding(encoder, secret, image)

            rescaled = (hidden_img[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            residual_viz = ((residual[0].permute(1,2,0).cpu().numpy() + 0.5) * 255).clip(0,255).astype(np.uint8)

            save_name = os.path.splitext(os.path.basename(filename))[0]
            Image.fromarray(rescaled).save(os.path.join(args.save_dir, save_name + "_hidden.png"))
            Image.fromarray(residual_viz).save(os.path.join(args.save_dir, save_name + "_residual.png"))


if __name__ == "__main__":
    main()


