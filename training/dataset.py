from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from training.conditions import CATEGORIES, encode_condition

PALETTES = (
    ((238, 92, 24), (255, 139, 35), (157, 45, 21)),
    ((78, 145, 65), (120, 179, 80), (35, 91, 55)),
    ((212, 188, 135), (239, 222, 171), (133, 115, 88)),
)
GLOWS = ((255, 211, 74), (117, 255, 116), (95, 192, 255), (205, 111, 255))


def labels_for_seed(seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    return {group: rng.randrange(len(values)) for group, values in CATEGORIES.items()}


def labels_for_index(index: int) -> dict[str, int]:
    """Enumerate every categorical combination before repeating any of them."""
    labels: dict[str, int] = {}
    for group, values in CATEGORIES.items():
        labels[group] = index % len(values)
        index //= len(values)
    return labels


def _star(cx: float, cy: float, outer: float, inner: float) -> list[tuple[float, float]]:
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        radius = outer if i % 2 == 0 else inner
        points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
    return points


def _eye_points(kind: int, cx: int, cy: int, side: int) -> list[tuple[float, float]]:
    if kind == 0:
        return [(cx - 6, cy + 4), (cx + 6, cy + 3), (cx + side * 2, cy - 6)]
    if kind == 2:
        return _star(cx, cy, 7, 3)
    return []


def render_pumpkin(labels: dict[str, int], seed: int, size: int = 128) -> Image.Image:
    """Render an original labeled pixel-art pumpkin without external assets."""
    rng = random.Random(seed ^ 0xA11CE)
    base = 64
    bg = (14 + rng.randrange(7), 8 + rng.randrange(5), 30 + rng.randrange(10))
    image = Image.new("RGB", (base, base), bg)
    draw = ImageDraw.Draw(image)

    # Night sky and ground create enough variation without overwhelming the subject.
    for _ in range(rng.randrange(5, 12)):
        x, y = rng.randrange(base), rng.randrange(4, 38)
        shade = rng.choice(((83, 70, 117), (247, 210, 119), (98, 92, 142)))
        draw.point((x, y), fill=shade)
    draw.rectangle((0, 51, 63, 63), fill=(12, 18 + rng.randrange(8), 24))
    draw.ellipse((9, 51, 55, 59), fill=(7, 9, 17))

    shape = labels["shape"]
    if shape == 0:
        box = [12, 21, 52, 55]
    elif shape == 1:
        box = [16, 14, 48, 56]
    elif shape == 2:
        box = [7, 25, 57, 54]
    else:
        box = [9, 20, 55, 56]
    jitter = rng.randrange(-2, 3)
    box = [box[0] + jitter, box[1], box[2] + jitter, box[3]]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    width, height = x1 - x0, y1 - y0

    glow = GLOWS[labels["glow"]]
    halo = Image.new("RGBA", (base, base), (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    halo_draw.ellipse((x0 - 5, y0 - 4, x1 + 5, y1 + 5), fill=(*glow, 35))
    halo = halo.filter(ImageFilter.GaussianBlur(4))
    image = Image.alpha_composite(image.convert("RGBA"), halo).convert("RGB")
    draw = ImageDraw.Draw(image)

    dark, light, outline = PALETTES[labels["color"]]
    draw.ellipse(box, fill=outline, outline=(7, 8, 14), width=2)
    lobes = 5 if shape != 3 else 6
    for i in range(lobes):
        t = i / max(1, lobes - 1)
        offset = int((t - 0.5) * width * 0.7)
        lobe_width = int(width * (0.38 if shape != 3 else rng.uniform(0.28, 0.42)))
        lx = cx + offset
        inset = int(abs(t - 0.5) * 3)
        draw.ellipse(
            (lx - lobe_width // 2, y0 + inset, lx + lobe_width // 2, y1 - inset),
            fill=dark if i in (0, lobes - 1) else light,
            outline=outline,
        )
    draw.ellipse((cx - width // 5, y0 + 2, cx + width // 5, y1 - 2), fill=light, outline=dark)
    for _ in range(28):
        px = rng.randrange(x0 + 3, x1 - 2)
        py = rng.randrange(y0 + 3, y1 - 2)
        if ((px - cx) / (width / 2)) ** 2 + ((py - cy) / (height / 2)) ** 2 < 0.88:
            draw.point((px, py), fill=rng.choice((dark, light, outline)))

    # Stem and accessory.
    draw.polygon([(cx - 3, y0 + 2), (cx - 2, y0 - 7), (cx + 4, y0 - 9), (cx + 3, y0 + 3)], fill=(55, 91, 48), outline=(20, 39, 26))
    accessory = labels["accessory"]
    if accessory == 1:
        draw.polygon([(cx - 14, y0 - 4), (cx + 15, y0 - 4), (cx + 2, y0 - 22)], fill=(61, 32, 91), outline=(15, 10, 24))
        draw.rectangle((cx - 17, y0 - 5, cx + 16, y0 - 2), fill=(32, 17, 51))
        draw.rectangle((cx - 3, y0 - 10, cx + 7, y0 - 7), fill=(237, 165, 45))
    elif accessory == 2:
        draw.pieslice((cx - 17, y0 - 15, cx + 17, y0 + 4), 180, 360, fill=(35, 29, 43), outline=(8, 7, 12), width=2)
        draw.line((cx - 16, y0 - 4, cx + 16, y0 - 4), fill=(185, 47, 39), width=2)
        draw.ellipse((cx - 2, y0 - 10, cx + 2, y0 - 6), fill=(231, 222, 184))
    elif accessory == 3:
        draw.ellipse((cx + 2, y0 - 8, cx + 13, y0 - 2), fill=(78, 150, 66), outline=(24, 67, 38))
        draw.line((cx + 3, y0 - 3, cx + 12, y0 - 7), fill=(24, 67, 38))

    # Carved eyes.
    expression = labels["expression"]
    eye_kind = labels["eyes"]
    eye_y = cy - max(5, height // 7)
    eye_dx = max(8, width // 5)
    for side in (-1, 1):
        ex = cx + side * eye_dx
        ey = eye_y + (side * 2 if expression == 1 else 0)
        if eye_kind in (0, 2):
            points = _eye_points(eye_kind, ex, ey, side)
            draw.polygon(points, fill=glow, outline=(50, 26, 16))
        elif eye_kind == 1:
            draw.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), fill=glow, outline=(50, 26, 16))
            draw.ellipse((ex - 1, ey - 1, ex + 2, ey + 2), fill=(71, 37, 24))
        else:
            slope = -side if expression == 3 else side
            draw.polygon([(ex - 6, ey), (ex + 6, ey + slope), (ex + 4, ey + 4), (ex - 5, ey + 3)], fill=glow, outline=(50, 26, 16))

    # Nose.
    draw.polygon([(cx, cy - 1), (cx - 3, cy + 5), (cx + 4, cy + 5)], fill=glow, outline=(50, 26, 16))

    mouth = labels["mouth"]
    my = cy + max(8, height // 5)
    if mouth == 0:
        draw.pieslice((cx - 12, my - 7, cx + 12, my + 7), 0, 180, fill=glow, outline=(50, 26, 16))
    elif mouth == 1:
        points = [(cx - 13, my - 2), (cx - 8, my + 4), (cx - 2, my), (cx + 4, my + 5), (cx + 13, my - 3), (cx + 9, my + 7), (cx - 10, my + 7)]
        draw.polygon(points, fill=glow, outline=(50, 26, 16))
    else:
        draw.rounded_rectangle((cx - 13, my - 4, cx + 13, my + 7), radius=3, fill=glow, outline=(50, 26, 16))
        if mouth == 2:
            for tx in (-7, 0, 7):
                draw.polygon([(cx + tx - 2, my - 4), (cx + tx + 2, my - 4), (cx + tx, my + 1)], fill=(53, 29, 20))
        else:
            draw.polygon([(cx - 8, my - 4), (cx - 3, my - 4), (cx - 5, my + 4)], fill=(244, 232, 192))
            draw.polygon([(cx + 3, my - 4), (cx + 8, my - 4), (cx + 5, my + 4)], fill=(244, 232, 192))

    if expression == 4:
        draw.arc((x0 + 4, y0 + 5, x1 - 4, y1 - 2), 200, 340, fill=(61, 21, 18), width=1)

    return image.resize((size, size), Image.Resampling.NEAREST)


@dataclass(frozen=True)
class PumpkinSample:
    image: np.ndarray
    condition: np.ndarray


class PumpkinDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, count: int, seed: int = 20261031, cache: bool = True, size: int = 64, balanced: bool = True) -> None:
        self.count = count
        self.seed = seed
        self.size = size
        self.balanced = balanced
        self.samples: list[PumpkinSample] | None = None
        if cache:
            self.samples = [self._make_sample(index) for index in range(count)]

    def _make_sample(self, index: int) -> PumpkinSample:
        sample_seed = self.seed + index * 7919
        labels = labels_for_index(index) if self.balanced else labels_for_seed(sample_seed)
        image = np.asarray(render_pumpkin(labels, sample_seed, size=self.size), dtype=np.uint8).copy()
        condition = np.asarray(encode_condition(labels), dtype=np.float32)
        return PumpkinSample(image=image, condition=condition)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index] if self.samples is not None else self._make_sample(index)
        image = torch.from_numpy(sample.image).permute(2, 0, 1).float().div(127.5).sub(1.0)
        return image, torch.from_numpy(sample.condition)

