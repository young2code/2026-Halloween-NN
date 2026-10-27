import torch

from training.conditions import CONDITION_DIM
from training.models import IMAGE_SIZE, LATENT_DIM, Discriminator, Generator


def test_model_shapes_and_parameter_budget() -> None:
    generator = Generator().eval()
    condition = torch.zeros(2, CONDITION_DIM)
    output = generator(torch.randn(2, LATENT_DIM), condition)
    assert output.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert sum(parameter.numel() for parameter in generator.parameters()) < 3_000_000
    discriminator = Discriminator().eval()
    score, logits = discriminator(output, condition)
    assert score.shape == (2, 1)
    assert logits.shape == (2, CONDITION_DIM)

