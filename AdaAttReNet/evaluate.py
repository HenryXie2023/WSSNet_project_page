import argparse

import torch

from datasets.sar_dataset import load_dataset
from losses.denoising_loss import build_loss
from models.ada_att_re_net import AdaAttReNet
from train import validate
from utils.common import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if config["training"]["cuda"] and torch.cuda.is_available() else "cpu")
    loader = load_dataset(config["data"]["val_dir"], config["data"]["val_size"], config, shuffled=False)
    model = AdaAttReNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    loss_fn = build_loss(config["training"]["loss"], is_mc=config["noise"]["type"] == "mc").to(device)
    valid_loss, valid_time, valid_psnr = validate(model, loss_fn, loader, device)
    print(f"valid_time={valid_time} valid_loss={valid_loss:>1.5f} valid_psnr={valid_psnr:.2f}")


if __name__ == "__main__":
    main()
