import numpy as np

from training.conditions import CONDITION_DIM, CATEGORIES, encode_condition
from training.dataset import PumpkinDataset, labels_for_index, labels_for_seed, render_pumpkin


def test_condition_vector_has_one_value_per_group() -> None:
    labels = labels_for_seed(42)
    vector = encode_condition(labels)
    assert len(vector) == CONDITION_DIM
    assert sum(vector) == len(CATEGORIES)


def test_renderer_is_deterministic_and_sized() -> None:
    labels = labels_for_seed(123)
    first = np.asarray(render_pumpkin(labels, 123))
    second = np.asarray(render_pumpkin(labels, 123))
    assert first.shape == (128, 128, 3)
    assert np.array_equal(first, second)


def test_training_dataset_is_native_size_and_balanced() -> None:
    dataset = PumpkinDataset(20, cache=False)
    image, _ = dataset[0]
    assert image.shape == (3, 64, 64)
    assert len({tuple(labels_for_index(i).values()) for i in range(20)}) == 20

