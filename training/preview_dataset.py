from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from training.conditions import describe
from training.dataset import labels_for_seed, render_pumpkin


def main() -> None:
    output = Path("docs/dataset-samples.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    cell, caption = 128, 25
    canvas = Image.new("RGB", (cell * 4, (cell + caption) * 4), (13, 8, 24))
    draw = ImageDraw.Draw(canvas)
    for index in range(16):
        seed = 20261031 + index * 7919
        labels = labels_for_seed(seed)
        x, y = (index % 4) * cell, (index // 4) * (cell + caption)
        canvas.paste(render_pumpkin(labels, seed), (x, y))
        draw.text((x + 3, y + cell + 3), describe(labels)[:25], fill=(235, 218, 180))
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
