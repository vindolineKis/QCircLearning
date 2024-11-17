import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from typing import List, Callable

class TrainerModel(nn.Module):
    def __init__(self, layers: List[nn.Module] = None, name: str = None):
        super(TrainerModel, self).__init__()
        self.model = nn.Sequential(*layers) if layers else nn.Sequential()
        self.name = name or 'TrainerModel'
        self.double()  # Use double precision for consistency

    def forward(self, x):
        return self.model(x)

    def back_minimize(self, x0: np.ndarray = None, method: str = 'L-BFGS-B', verbose: int = 0):
        x_list, f_list, gradient_list = [], [], []

        def to_minimize(x):
            x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            return self(x_tensor).detach().numpy()

        x = x0 if x0 is not None else np.random.rand(self.model[0].in_features)

        def to_minimize_with_grad(x):
            x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            x_list.append(x)
            loss = self(x_tensor)
            f_list.append(loss.item())
            loss.backward()
            gradients = x_tensor.grad.numpy()
            gradient_list.append(gradients)
            return loss.item(), gradients

        result = minimize(to_minimize_with_grad, x, bounds=[(-np.pi * 2, np.pi * 2)] * len(x), jac=True, method=method, tol=1e-6,
                          options={'disp': None, 'maxls': 20, 'iprint': -1, 'eps': 1e-7, 'ftol': 1e-6, 'maxiter': 1500, 'maxcor': 12, 'maxfun': 1500})

        if verbose:
            print("Optimization result:", result)
            print(f'Optimization converged: {result.success}')
            print(f'Stored x_list: {x_list[-1]}')
            print(f'Stored f_list: {f_list}')
            print(f'Stored gradient_list: {gradient_list[-1]}')
            if result.success:
                print("Optimization converged successfully.")
            else:
                print("Optimization did not converge. Reason:", result.message)

        return result.x

    @staticmethod
    def default_model(input_shape: tuple):
        return TrainerModel(
            layers=[
                nn.Linear(input_shape[0], 96),
                nn.ELU(),
                nn.Linear(96, 64),
                nn.ELU(),
                nn.Linear(64, 18),
                nn.ELU(),
                nn.Linear(18, 10),
                nn.ELU(),
                nn.Linear(10, 1)
            ],
            name='default_model'
        )

    @staticmethod
    def simple_model(input_shape: tuple):
        return TrainerModel(
            layers=[
                nn.Linear(input_shape[0], 32),
                nn.ELU(),
                nn.Linear(32, 8),
                nn.Sigmoid(),
                nn.Linear(8, 1)
            ],
            name='simple_model'
        )
