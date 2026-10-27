from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from training.conditions import CATEGORIES, CONDITION_DIM, describe, encode_condition
from training.dataset import PumpkinDataset
from training.models import LATENT_DIM, Discriminator, Generator, initialize_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tiny Pumpkin conditional GAN.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--output", type=Path, default=Path("training/runs/main"))
    parser.add_argument("--sample-every", type=int, default=5)
    return parser.parse_args()


def fixed_conditions() -> tuple[torch.Tensor, list[str]]:
    labels_list: list[dict[str, int]] = []
    for i in range(16):
        labels_list.append({group: i % len(values) for group, values in CATEGORIES.items()})
    tensor = torch.tensor([encode_condition(labels) for labels in labels_list], dtype=torch.float32)
    return tensor, [describe(labels) for labels in labels_list]


def save_grid(images: torch.Tensor, captions: list[str], path: Path) -> None:
    images = images.detach().cpu().clamp(-1, 1).add(1).mul(127.5).byte().permute(0, 2, 3, 1).numpy()
    cell, caption_height = 128, 24
    canvas = Image.new("RGB", (cell * 4, (cell + caption_height) * 4), (13, 8, 24))
    draw = ImageDraw.Draw(canvas)
    for i, pixels in enumerate(images):
        x, y = (i % 4) * cell, (i // 4) * (cell + caption_height)
        canvas.paste(Image.fromarray(pixels), (x, y))
        draw.text((x + 3, y + cell + 2), captions[i][:24], fill=(235, 218, 180))
    canvas.save(path)


@torch.no_grad()
def update_ema(ema: nn.Module, source: nn.Module, decay: float = 0.995) -> None:
    ema_params = dict(ema.named_parameters())
    for name, parameter in source.named_parameters():
        ema_params[name].lerp_(parameter.detach(), 1 - decay)
    ema_buffers = dict(ema.named_buffers())
    for name, buffer in source.named_buffers():
        ema_buffers[name].copy_(buffer)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the planned training run.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"Preparing {args.samples} deterministic procedural samples...", flush=True)
    dataset = PumpkinDataset(args.samples, seed=args.seed, cache=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    conditions, captions = fixed_conditions()
    conditions = conditions.to(device)
    fixed_latent = torch.randn(len(conditions), LATENT_DIM, generator=torch.Generator().manual_seed(args.seed)).to(device)
    history: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        generator.train()
        d_total = g_total = 0.0
        for real, condition in loader:
            real = real.to(device, non_blocking=True)
            condition = condition.to(device, non_blocking=True)
            batch = real.shape[0]

            optimizer_d.zero_grad(set_to_none=True)
            latent = torch.randn(batch, LATENT_DIM, device=device)
            fake = generator(latent, condition)
            real_score = discriminator(real, condition)
            fake_score = discriminator(fake.detach(), condition)
            loss_d = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            latent = torch.randn(batch, LATENT_DIM, device=device)
            fake = generator(latent, condition)
            loss_g = -discriminator(fake, condition).mean()
            loss_g.backward()
            optimizer_g.step()
            update_ema(ema, generator)
            d_total += float(loss_d.detach())
            g_total += float(loss_g.detach())

        record = {
            "epoch": epoch,
            "loss_d": d_total / len(loader),
            "loss_g": g_total / len(loader),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            ema.eval()
            with torch.no_grad():
                sample = ema(fixed_latent, conditions)
            save_grid(sample, captions, args.output / f"samples-{epoch:03d}.png")
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "generator_ema": ema.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "config": vars(args) | {"output": str(args.output), "condition_dim": CONDITION_DIM, "latent_dim": LATENT_DIM},
                },
                args.output / "checkpoint.pt",
            )
            (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

