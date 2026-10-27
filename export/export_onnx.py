from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from training.conditions import CONDITION_DIM
from training.models import LATENT_DIM, Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and verify the trained pumpkin generator.")
    parser.add_argument("--checkpoint", type=Path, default=Path("training/runs/main/checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("public/generator.onnx"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = Generator().eval()
    model.load_state_dict(checkpoint["generator_ema"])
    torch.manual_seed(1234)
    latent = torch.randn(1, LATENT_DIM)
    condition = torch.zeros(1, CONDITION_DIM)
    offset = 0
    for length in (4, 5, 4, 4, 3, 4, 4):
        condition[0, offset] = 1
        offset += length
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (latent, condition),
        args.output,
        input_names=["latent", "condition"],
        output_names=["image"],
        dynamic_axes={"latent": {0: "batch"}, "condition": {0: "batch"}, "image": {0: "batch"}},
        opset_version=17,
    )
    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"latent": latent.numpy(), "condition": condition.numpy()})[0]
    expected = model(latent, condition).detach().numpy()
    error = float(np.max(np.abs(actual - expected)))
    size_mb = args.output.stat().st_size / 1024 / 1024
    if error > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max absolute error {error}")
    if size_mb > 15:
        raise RuntimeError(f"Exported generator is too large: {size_mb:.2f} MB")
    print(f"Exported {args.output} ({size_mb:.2f} MB); max error={error:.8f}")


if __name__ == "__main__":
    main()

