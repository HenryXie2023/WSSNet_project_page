import argparse
import os

import torch
import torchvision.transforms.functional as tvF
from PIL import Image

from models.ada_att_re_net import AdaAttReNet
from utils.common import load_config, pad_to_multiple, tensor_to_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    input_dir = args.input_dir or config["data"]["test_dir"]
    output_dir = args.output_dir or config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if config["training"]["cuda"] and torch.cuda.is_available() else "cpu")
    model = AdaAttReNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    image_names = sorted(
        name
        for name in os.listdir(input_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    )

    with torch.no_grad():
        for name in image_names:
            image = Image.open(os.path.join(input_dir, name)).convert("RGB")
            tensor = tvF.to_tensor(image).unsqueeze(0).to(device)
            padded, height, width = pad_to_multiple(tensor, multiple=32)
            output = model(padded)[:, :, :height, :width]
            tensor_to_image(output.squeeze(0)).save(os.path.join(output_dir, name))


if __name__ == "__main__":
    main()
