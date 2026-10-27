from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from training.conditions import CONDITION_DIM

LATENT_DIM = 96
IMAGE_SIZE = 64


class UpsampleBlock(nn.Module):
    """Nearest-neighbor resize followed by convolution avoids checkerboard artifacts."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, condition_dim: int = CONDITION_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.project = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            UpsampleBlock(512, 256),
            UpsampleBlock(256, 128),
            UpsampleBlock(128, 64),
            UpsampleBlock(64, 32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.project(torch.cat((latent, condition), dim=1)))


class Discriminator(nn.Module):
    def __init__(self, condition_dim: int = CONDITION_DIM) -> None:
        super().__init__()
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.score = spectral_norm(nn.Linear(512, 1))
        self.condition_embedding = spectral_norm(nn.Linear(condition_dim, 512, bias=False))
        self.condition_classifier = spectral_norm(nn.Linear(512, condition_dim))

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(image)
        projection = (features * self.condition_embedding(condition)).sum(dim=1, keepdim=True) / math.sqrt(512)
        return self.score(features) + projection, self.condition_classifier(features)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)

