# Procedural Pumpkin Dataset — Dataset Card

## Overview

The training set is generated locally by `training/dataset.py`. It uses Pillow
to draw original 64×64 pixel-art scenes at the model's native resolution.
No photographs, scraped images, generated images
from an outside model, or personal information are used.

## Labels

Every sample contains one value from each group:

- 4 shapes
- 5 expressions
- 4 eye styles
- 4 mouth styles
- 3 pumpkin colors
- 4 glow colors
- 4 accessories

The training sampler enumerates all 15,360 categorical combinations before it
repeats one, while each integer seed adds deterministic visual variation. This makes
the dataset reproducible without storing thousands of image files in Git.

## Provenance and license

The renderer and dataset recipe were created for this repository and are
released under the repository's MIT License. Contributors must not add artwork
unless they own it and document its provenance and license here.

## Known biases and limitations

The dataset intentionally represents a narrow, whimsical Western Halloween
visual vocabulary. It is not a survey of cultural pumpkin traditions. Its
balanced categorical sampling does not guarantee that every combination is
equally easy for the model to learn.

