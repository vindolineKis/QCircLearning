# test_back_minimizer.py
import pytest
import torch
import numpy as np
from .trainerModel import TrainerModel
from .back_minimizer import BackMinimizer
from scipy.optimize import OptimizeResult


@pytest.fixture
def model():
    """Fixture to create a default TrainerModel."""
    return TrainerModel.default_model((5,)).double()


@pytest.fixture
def minimizer(model):
    """Fixture to create a BackMinimizer with the given model."""
    return BackMinimizer(model)


def test_gradient_alignment_finite_difference(minimizer):
    x = np.random.uniform(-2 * np.pi, 2 * np.pi, 5)
    loss, gradients = minimizer.to_minimize_with_grad(x)
    # Numerical gradient via finite differences
    epsilon = 1e-5
    numerical_gradients = np.zeros_like(x)
    for i in range(len(x)):
        x_eps_plus = np.copy(x)
        x_eps_minus = np.copy(x)
        x_eps_plus[i] += epsilon
        x_eps_minus[i] -= epsilon
        
        loss_plus, _ = minimizer.to_minimize_with_grad(x_eps_plus)
        loss_minus, _ = minimizer.to_minimize_with_grad(x_eps_minus)
        
        numerical_gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)

    # Compare analytical gradients with numerical gradients
    np.testing.assert_allclose(gradients, numerical_gradients, rtol=1e-4, atol=1e-6)
    

def test_gradient_norm_during_training(minimizer):

    x0 = np.random.uniform(-2 * np.pi, 2 * np.pi, 5)
    minimizer.back_minimize(x0, method="L-BFGS-B", verbose=0)
    
    for gradient in minimizer.gradient_list:
        grad_norm = np.linalg.norm(gradient)
        assert grad_norm > 0, "Gradient norm should be positive"
        assert grad_norm > 1e-3, "Gradient norm should be large enough"

    # Optionally, check if the norms are decreasing (if applicable)
    # for i in range(1, len(gradient_norms)):
    #     assert gradient_norms[i] <= gradient_norms[i-1], "Gradient norm should decrease during training"




