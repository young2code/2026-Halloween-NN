import torch

from training.conditions import CONDITION_DIM
from training.models import LATENT_DIM, Discriminator, Generator


def test_model_shapes_and_parameter_budget() -> None:
    generator = Generator().eval()
    condition = torch.zeros(2, CONDITION_DIM)
    output = generator(torch.randn(2, LATENT_DIM), condition)
    assert output.shape == (2, 3, 128, 128)
    assert sum(parameter.numel() for parameter in generator.parameters()) < 1_500_000
    discriminator = Discriminator().eval()
    assert discriminator(output, condition).shape == (2, 1)

