# Tiny Pumpkin Machine — Model Card

## Summary

Tiny Pumpkin Machine is a small conditional generative adversarial network
(cGAN) that produces 128×128 illustrated pixel-art jack-o'-lanterns. It is
designed for teaching and browser inference, not photorealism.

## Inputs and output

- `latent`: 96 normally distributed random values.
- `condition`: a 28-value vector containing one selected value from each of
  seven groups: shape, expression, eyes, mouth, color, glow, and accessory.
- `image`: an RGB tensor with shape `1 × 3 × 128 × 128` and values in `[-1, 1]`.

The browser converts a controlled English prompt into the condition vector.
Words outside the documented vocabulary are ignored and shown to the user.

## Architecture

The generator projects the concatenated latent and condition vectors into a
4×4 feature map, then applies five transposed-convolution stages. Batch
normalization and ReLU activations are used between stages, with `tanh` at the
output. The discriminator is a projection discriminator with spectrally
normalized convolution and linear layers. Only the generator is exported.

The released EMA generator has 1,213,667 trainable parameters and occupies
4.67 MiB in ONNX format. PyTorch and ONNX Runtime outputs differed by at most
0.00000167 in the fixed export-parity test.

## Training

- Hardware: NVIDIA GeForce RTX 4070 SUPER with 12 GB VRAM.
- Dataset: 6,000 deterministic procedural samples.
- Schedule: 45 epochs, batch size 64, Adam with learning rate 0.0002.
- Objective: hinge adversarial loss with a projection discriminator.
- Stabilization: spectral normalization and generator EMA with decay 0.995.
- Wall time: approximately 142 seconds on the training hardware.

The browser smoke test generated on the WASM CPU path in approximately 0.02
seconds in headless Chrome. Performance varies by device and browser.

## Intended use

- Family-friendly creative play.
- Demonstrating conditional generative models.
- A compact ONNX Runtime Web example.
- Experiments with controlled prompts, random seeds, and latent interpolation.

## Out-of-scope use

This model should not be used for realistic imagery, identity depiction,
high-stakes decisions, or general text-to-image generation. It only knows the
small set of pumpkin attributes on which it was trained.

## Limitations

- GANs can repeat patterns or collapse to a small set of outputs.
- Conditions may occasionally blend or be ignored.
- Procedural training data limits the model to the renderer's visual world.
- Output is deliberately low resolution.
- A prompt parser is not equivalent to natural-language understanding.

## Privacy

The released model runs on the visitor's device through WebGPU or WASM. Prompt,
seed, condition values, and image are not sent to an inference service.

