import argparse
import json
import os
from datetime import datetime

import torch
from torch.optim import Adam

from datasets.sar_dataset import load_dataset
from losses.denoising_loss import build_loss, denoising_loss
from models.ada_att_re_net import AdaAttReNet
from utils.common import AvgMeter, load_config, psnr, seed_all, time_elapsed_since


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def adjust_learning_rate(optimizer, iteration, total_iterations):
    ramp_down_perc = 0.3
    initial_lr = 0.001
    ramp_down_start_iter = total_iterations * (1 - ramp_down_perc)
    if iteration >= ramp_down_start_iter:
        t = (iteration - ramp_down_start_iter) / (ramp_down_perc * total_iterations)
        lr = initial_lr * (0.5 + torch.cos(torch.tensor(t * torch.pi)).item() / 2) ** 2
    else:
        lr = initial_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def validate(model, loss_fn, valid_loader, device):
    model.eval()
    loss_meter = AvgMeter()
    psnr_meter = AvgMeter()
    valid_start = datetime.now()

    with torch.no_grad():
        for source, target in valid_loader:
            source = source.to(device)
            target = target.to(device)
            output = model(source)
            loss = loss_fn(output, target)
            loss_meter.update(loss.item())
            for i in range(source.size(0)):
                psnr_meter.update(psnr(output[i].cpu(), target[i].cpu()).item())

    valid_time = time_elapsed_since(valid_start)[0]
    return loss_meter.avg, valid_time, psnr_meter.avg


def save_checkpoint(model, output_dir, noise_type, epoch, stats, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    if epoch == 0:
        ckpt_dir = os.path.join(output_dir, noise_type if overwrite else f"{noise_type}-{datetime.now():%H%M}")
        os.makedirs(ckpt_dir, exist_ok=True)
        save_checkpoint.ckpt_dir = ckpt_dir

    if overwrite:
        filename = os.path.join(save_checkpoint.ckpt_dir, f"checkpoint-{noise_type}.pt")
    else:
        valid_loss = stats["valid_loss"][epoch]
        filename = os.path.join(save_checkpoint.ckpt_dir, f"checkpoint-epoch{epoch + 1}-{valid_loss:>1.5f}.pt")

    torch.save(model.state_dict(), filename)
    with open(os.path.join(save_checkpoint.ckpt_dir, "stats.json"), "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)


def train(config):
    seed_all(config["training"]["seed"])
    device = torch.device("cuda" if config["training"]["cuda"] and torch.cuda.is_available() else "cpu")

    train_loader = load_dataset(config["data"]["train_dir"], config["data"]["train_size"], config, shuffled=True)
    valid_loader = load_dataset(config["data"]["val_dir"], config["data"]["val_size"], config, shuffled=False)

    model = AdaAttReNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
    ).to(device)
    optimizer = Adam(
        list(model.parameters()),
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["adam"][:2]),
        eps=config["training"]["adam"][2],
    )
    loss_fn = build_loss(config["training"]["loss"], is_mc=config["noise"]["type"] == "mc").to(device)

    num_batches = len(train_loader)
    total_iterations = config["training"]["epochs"] * num_batches
    assert num_batches % config["training"]["report_interval"] == 0, "report_interval must divide total batches"

    stats = {
        "noise_type": config["noise"]["type"],
        "noise_param": config["noise"]["param"],
        "train_loss": [],
        "valid_loss": [],
        "valid_psnr": [],
    }

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss_meter = AvgMeter()
        loss_meter = AvgMeter()
        epoch_start = datetime.now()

        for batch_idx, (source, target) in enumerate(train_loader):
            iteration = epoch * num_batches + batch_idx
            adjust_learning_rate(optimizer, iteration, total_iterations)
            source = source.to(device)
            target = target.to(device)

            loss, _ = denoising_loss(loss_fn, model, source, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            if (batch_idx + 1) % config["training"]["report_interval"] == 0 and batch_idx:
                train_loss_meter.update(loss_meter.avg)
                loss_meter.reset()

        valid_loss, valid_time, valid_psnr = validate(model, loss_fn, valid_loader, device)
        stats["train_loss"].append(train_loss_meter.avg)
        stats["valid_loss"].append(valid_loss)
        stats["valid_psnr"].append(valid_psnr)
        save_checkpoint(
            model,
            config["training"]["output_dir"],
            config["noise"]["type"],
            epoch,
            stats,
            overwrite=config["training"]["checkpoint_overwrite"],
        )
        epoch_time = time_elapsed_since(epoch_start)[0]
        print(
            f"epoch={epoch + 1} train_time={epoch_time} valid_time={valid_time} "
            f"valid_loss={valid_loss:>1.5f} valid_psnr={valid_psnr:.2f}"
        )


if __name__ == "__main__":
    args = parse_args()
    train(load_config(args.config))
