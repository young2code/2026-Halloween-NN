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
from training.models import IMAGE_SIZE, LATENT_DIM, Discriminator, Generator, initialize_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tiny Pumpkin conditional GAN.")
    parser.add_argument("--samples", type=int, default=61440)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--output", type=Path, default=Path("training/runs/main"))
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--max-hours", type=float, default=5.5)
    parser.add_argument("--resume", type=Path)
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
        image = Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        draw.text((x + 3, y + cell + 2), captions[i][:24], fill=(235, 218, 180))
    canvas.save(path)


@torch.no_grad()
def update_ema(ema: nn.Module, source: nn.Module, decay: float = 0.999) -> None:
    ema_params = dict(ema.named_parameters())
    for name, parameter in source.named_parameters():
        ema_params[name].lerp_(parameter.detach(), 1 - decay)
    ema_buffers = dict(ema.named_buffers())
    for name, buffer in source.named_buffers():
        ema_buffers[name].copy_(buffer)


def categorical_loss(logits: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    offset = 0
    for values in CATEGORIES.values():
        width = len(values)
        targets = condition[:, offset : offset + width].argmax(dim=1)
        losses.append(F.cross_entropy(logits[:, offset : offset + width], targets))
        offset += width
    return torch.stack(losses).mean()


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


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
    torch.set_float32_matmul_precision("high")

    print(f"Preparing {args.samples} balanced {IMAGE_SIZE}x{IMAGE_SIZE} procedural samples...", flush=True)
    dataset = PumpkinDataset(args.samples, seed=args.seed, cache=True, size=IMAGE_SIZE, balanced=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    ema = copy.deepcopy(generator).eval().requires_grad_(False)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()
    conditions, captions = fixed_conditions()
    conditions = conditions.to(device)
    fixed_latent = torch.randn(len(conditions), LATENT_DIM, generator=torch.Generator().manual_seed(args.seed)).to(device)
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        ema.load_state_dict(checkpoint["generator_ema"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        if "scaler_g" in checkpoint:
            scaler_g.load_state_dict(checkpoint["scaler_g"])
            scaler_d.load_state_dict(checkpoint["scaler_d"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming after epoch {start_epoch - 1}.", flush=True)

    generator_parameters = sum(parameter.numel() for parameter in generator.parameters())
    print(f"Generator parameters: {generator_parameters:,}; batches per epoch: {len(loader)}", flush=True)
    started = time.time()

    def save_checkpoint(epoch: int) -> None:
        config = vars(args) | {
            "output": str(args.output),
            "resume": str(args.resume) if args.resume else None,
            "condition_dim": CONDITION_DIM,
            "latent_dim": LATENT_DIM,
            "image_size": IMAGE_SIZE,
        }
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "generator_ema": ema.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scaler_g": scaler_g.state_dict(),
                "scaler_d": scaler_d.state_dict(),
                "history": history,
                "config": config,
            },
            args.output / "checkpoint.pt",
        )
        torch.save(
            {"epoch": epoch, "generator_ema": ema.state_dict(), "config": config},
            args.output / f"generator-{epoch:03d}.pt",
        )
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        d_total = g_total = aux_total = 0.0
        for real, condition in loader:
            real = real.to(device, non_blocking=True)
            condition = condition.to(device, non_blocking=True)
            batch = real.shape[0]

            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                latent = torch.randn(batch, LATENT_DIM, device=device)
                fake = generator(latent, condition)
                real_score, real_logits = discriminator(real, condition)
                fake_score, _ = discriminator(fake.detach(), condition)
                wrong_score, _ = discriminator(real, condition.roll(1, dims=0))
                auxiliary = categorical_loss(real_logits, condition)
                loss_d = (
                    F.relu(1 - real_score).mean()
                    + 0.5 * F.relu(1 + fake_score).mean()
                    + 0.5 * F.relu(1 + wrong_score).mean()
                    + 0.5 * auxiliary
                )
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                latent = torch.randn(batch, LATENT_DIM, device=device)
                fake = generator(latent, condition)
                fake_score, fake_logits = discriminator(fake, condition)
                fake_auxiliary = categorical_loss(fake_logits, condition)
                loss_g = -fake_score.mean() + args.aux_weight * fake_auxiliary
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            update_ema(ema, generator)
            d_total += float(loss_d.detach())
            g_total += float(loss_g.detach())
            aux_total += float(fake_auxiliary.detach())

        elapsed = time.time() - started
        record = {
            "epoch": epoch,
            "loss_d": d_total / len(loader),
            "loss_g": g_total / len(loader),
            "loss_g_attributes": aux_total / len(loader),
            "elapsed_seconds": elapsed,
            "images_per_second": (epoch - start_epoch + 1) * len(loader) * args.batch_size / elapsed,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        final_epoch = epoch
        timed_out = elapsed >= args.max_hours * 3600
        if epoch == start_epoch or epoch % args.sample_every == 0 or epoch == args.epochs or timed_out:
            ema.eval()
            with torch.no_grad():
                sample = ema(fixed_latent, conditions)
            save_grid(sample, captions, args.output / f"samples-{epoch:03d}.png")
            save_checkpoint(epoch)
        if timed_out:
            print(f"Reached the {args.max_hours:.2f}-hour safety limit after epoch {epoch}.", flush=True)
            break

    if final_epoch >= start_epoch and final_epoch % args.sample_every:
        ema.eval()
        with torch.no_grad():
            sample = ema(fixed_latent, conditions)
        save_grid(sample, captions, args.output / f"samples-{final_epoch:03d}.png")
        save_checkpoint(final_epoch)


if __name__ == "__main__":
    main()

