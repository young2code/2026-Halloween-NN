from __future__ import annotations

from collections.abc import Mapping

CATEGORIES: dict[str, tuple[str, ...]] = {
    "shape": ("round", "tall", "wide", "lumpy"),
    "expression": ("happy", "silly", "surprised", "angry", "spooky"),
    "eyes": ("triangle", "round eyes", "star eyes", "sleepy"),
    "mouth": ("smile", "crooked", "toothy", "fangs"),
    "color": ("orange", "green", "pale"),
    "glow": ("yellow glow", "green glow", "blue glow", "purple glow"),
    "accessory": ("no hat", "witch hat", "pirate hat", "leaf"),
}

CONDITION_DIM = sum(len(values) for values in CATEGORIES.values())


def encode_condition(labels: Mapping[str, int]) -> list[float]:
    vector: list[float] = []
    for group, values in CATEGORIES.items():
        selected = labels[group]
        if selected < 0 or selected >= len(values):
            raise ValueError(f"Invalid {group} value: {selected}")
        vector.extend(1.0 if i == selected else 0.0 for i in range(len(values)))
    return vector


def describe(labels: Mapping[str, int]) -> str:
    words = [CATEGORIES[group][labels[group]] for group in CATEGORIES]
    return ", ".join(words)

