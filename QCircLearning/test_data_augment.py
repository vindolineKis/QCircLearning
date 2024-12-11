import pytest
import numpy as np
from QCircLearning.utils import (
    data_augmentation,
    data_augment_noise,
    data_augment_periodic,
)


class MockBackMinimizer:
    def back_minimize(self, x0, method, **kwargs):
        return x0 - 0.1  # Mock behavior for testing


@pytest.fixture
def setup():
    data = np.array([1.0, 2.0, 3.0])
    config = {
        "noise_augment_points": 2,
        "periodic_augment_points": 2,
        "noise_strengh": 0.01,
    }
    backminimizer = MockBackMinimizer()
    return data, config, backminimizer


def mock_circ_evaluate(x):
    return np.sum(x**2)  # Mock behavior for testing


def test_data_augmentation(setup):
    data, config, backminimizer = setup
    new_data_x, new_data_y = data_augmentation(
        data, mock_circ_evaluate, backminimizer, config
    )

    assert len(new_data_x) == 9
    assert len(new_data_y) == 9
    assert isinstance(new_data_x, list)
    assert isinstance(new_data_y, list)


def test_data_augment_noise():
    data = np.array([1.0, 2.0, 3.0])
    noise_strength = 0.05
    augmented_data = data_augment_noise(data, noise_strength)
    assert augmented_data.shape == data.shape


def test_data_augment_periodic():
    data = np.array([0.0, np.pi, 2 * np.pi])
    n_points = 3
    shift = np.pi / 2
    augmented_data = data_augment_periodic(data, n_points, shift)
    assert len(augmented_data) == n_points + 1
    for d in augmented_data:
        assert len(d) == len(data)
        assert np.all(d >= data - shift) and np.all(d <= data + shift)


def test_data_augmentation_no_noise(setup):
    data, config, backminimizer = setup
    config["noise_augment_points"] = 0
    new_data_x, new_data_y = data_augmentation(
        data, mock_circ_evaluate, backminimizer, config
    )

    assert len(new_data_x) == 3
    assert len(new_data_y) == 3


def test_data_augmentation_no_periodic(setup):
    data, config, backminimizer = setup
    config["periodic_augment_points"] = 0
    new_data_x, new_data_y = data_augmentation(
        data, mock_circ_evaluate, backminimizer, config
    )

    assert len(new_data_x) == 3
    assert len(new_data_y) == 3
