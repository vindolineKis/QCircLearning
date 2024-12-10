# trainer_model.py
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import List, Callable


class BackMinimizer:
    def __init__(self, model: nn.Module):
        """
        Initializes the BackMinimizer with a given model.

        Parameters:
            model (nn.Module): The neural network model to optimize.
        """
        self.model = model
        self.model.eval()
        self.x_list = []
        self.f_list = []
        self.gradient_list = []

    def to_minimize(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        return self.model(x_tensor).detach().numpy()

    def to_minimize_with_grad(self, x: np.ndarray):
        """
        Computes the loss and gradients for the optimizer.

        Parameters:
            x (np.ndarray): Input array for optimization.

        Returns:
            tuple: Loss value and gradients.
        """
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        loss = self.model(x_tensor)
        loss.backward()
        gradients = x_tensor.grad.numpy()
        self.x_list.append(x)
        self.f_list.append(loss.item())
        self.gradient_list.append(gradients)

        return loss.item(), gradients

    def back_minimize(
        self,
        x0: np.ndarray = None,
        method: str = "L-BFGS-B",
        verbose: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """
        Performs optimization to minimize the model's loss.

        Parameters:
            x0 (np.ndarray, optional): Initial guess for the variables.
            method (str): Optimization method.
            verbose (int): Verbosity level.
            **kwargs: Additional keyword arguments for the optimizer.

        Returns:
            np.ndarray: Optimized variables.
        """

        def callback(xk):
            if verbose:
                print(f"Current x: {xk}")

        x0 = x0 if x0 is not None else np.random.rand(self.model.input_shape)
        result = minimize(
            self.to_minimize_with_grad,
            x0,
            bounds=[(-2 * np.pi, 2 * np.pi)] * x0.shape[0],
            jac=True,
            method=method,
            tol=1e-6,
            options=kwargs.get("back_minimize_options", {}),
            callback=callback,
        )

        if verbose:
            print("Optimization result:", result)
            print(f"Optimization converged: {result.success}")
            print(f"Stored x_list: {self.x_list[-1]}")
            print(f"Stored f_list: {self.f_list}")
            print(f"Stored gradient_list: {self.gradient_list[-1]}")
            if result.success:
                print("Optimization converged successfully.")
            else:
                print("Optimization did not converge. Reason:", result.message)

        return result.x
