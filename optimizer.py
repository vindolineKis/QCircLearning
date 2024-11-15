from typing import List
from scipy.optimize import minimize, OptimizeResult
from nn_trainer import TrainerModel
import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim


import sys
import os
import random

# Set the seed value for reproducibility
# SEED = 42

# # Set the PYTHONHASHSEED environment variable
# os.environ['PYTHONHASHSEED'] = str(SEED)

# # Set random seed for Python's `random` module
# random.seed(SEED)

# # Set random seed for NumPy
# np.random.seed(SEED)

# # Set random seed for PyTorch
# torch.manual_seed(SEED)

# Additional configuration for GPU determinism in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class Optimizer:

    _available_methods = ['Neural Network', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'linear_regression', 'random search', 'Adam']

    @staticmethod
    def list_methods():
        return Optimizer._available_methods

    def __init__(self, method: str = 'Neural Network') -> None:
        self.method = method
        self.saved_path = None

    @property
    def get_path_x(self):
        if self.saved_path is None:
            return None
        else:
            return self.path_x

    @property
    def get_path_y(self):
        if self.saved_path is None:
            return None
        else:
            return self.path_y

    def optimize(self,
                 func,
                 x0,
                 callback=None,
                 record_path: bool = True,
                 method: str = None,
                 **kwargs) -> OptimizeResult:

        if record_path:
            self.path_x = []
            self.path_y = []

            def min_func(x):
                self.path_x.append(x)
                y = func(x)
                self.path_y.append(y)
                return y
        else:
            def min_func(x):
                return func(x)

        if method is None:
            method = self.method

        if method not in Optimizer._available_methods:
            raise ValueError(f'Optimizer method {method} not available. Available methods are {self.list_methods()}')

        # For Neural Network, directly call _NN_opt instead of using minimize
        if method == 'Neural Network':
            return self._NN_opt(func=min_func, x0=x0, callback=callback, **kwargs)

        elif method == 'random search':
            return self._random_search(func=min_func, x0=x0, callback=callback, **kwargs)

        elif method in ['BFGS', 'Nelder-Mead', 'Powell', 'CG']:
            return minimize(min_func, x0, method=method, jac='3-point' if method == 'BFGS' else None, callback=callback, options=kwargs)

        else:
            raise ValueError(f'Unknown optimization method: {method}')

    def _NN_opt(self, func, x0, callback=None, **kwargs):
        # Optimize using neural network
        para_size = len(x0)
        res = OptimizeResult()
        res.nfev = 0
        res.nit = 0

        # Define the default values
        init_data = kwargs.get('init_data', np.random.uniform(-10, 10, (60, para_size)))
        max_iter = kwargs.get('max_iter', 20)
        classical_epochs = kwargs.get('classical_epochs', 20)
        batch_size = kwargs.get('batch_size', 16)
        verbose = kwargs.get('verbose', 0)
        nn_models = kwargs.get('NN_Models', [
            TrainerModel.default_model((para_size,)),
            TrainerModel.simple_model((para_size,))
        ])

        sample_y = np.array([])
        sample_x = np.empty((0, para_size))
        optimal = [None, float('inf')]  # [opt_para, opt_y]

        # Generate initial sample data
        for para in init_data:
            y = func(para)
            if y < optimal[1]:
                optimal = [para, y]
            sample_x = np.vstack([sample_x, para])
            sample_y = np.append(sample_y, y)

        print(f'Training with the neural networks')
        sys.stdout.flush()

        for model in nn_models:
            print(model)
            sys.stdout.flush()

            for iteration in range(max_iter):
                res.nit += 1
                print(f'Iteration {iteration + 1}/{max_iter}')

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-4)

                model.train()
                for epoch in range(classical_epochs):
                    total_loss = 0.0
                    permutation = torch.randperm(sample_x.shape[0])

                    for i in range(0, sample_x.shape[0], batch_size):
                        indices = permutation[i:i + batch_size]
                        batch_x, batch_y = sample_x[indices], sample_y[indices]

                        inputs = torch.tensor(batch_x, dtype=torch.float64)
                        targets = torch.tensor(batch_y, dtype=torch.float64).unsqueeze(-1)

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        loss = criterion(outputs, targets)
                        total_loss += loss.item()

                        loss.backward()
                        optimizer.step()

                    # Print average loss for the epoch if verbose
                    if verbose:
                        avg_loss = total_loss / (sample_x.shape[0] // batch_size)
                        
                        print(f'Epoch {epoch + 1}/{classical_epochs}, Average Loss: {avg_loss:1e}')
                        sys.stdout.flush()

                # Prediction and updating optimal parameters
                x0 = optimal[0]
                prediction0 = model.back_minimize(x0=x0, method='L-BFGS-B', verbose=verbose)
                y0 = func(prediction0)
                res.nfev += 1

                if y0 < optimal[1]:
                    optimal = [prediction0, y0]

                predictions = np.vstack([prediction0, prediction0 + np.pi * 2, prediction0 - np.pi * 2])
                sample_x = np.concatenate([sample_x, predictions], axis=0)
                sample_y = np.concatenate([sample_y, [y0] * predictions.shape[0]])

        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])

        return res


    def _random_search(self, func, x0, callback=None, **kwargs):
        para_size = len(x0)
        res = OptimizeResult()
        res.nfev = 0
        res.nit = 0

        init_data = kwargs.get('init_data', np.random.uniform(-10, 10, (60, para_size)))
        max_iter = kwargs.get('max_iter', 20)

        sample_x = init_data
        sample_y = np.array([])

        optimal = [None, 1]  # [opt_para, opt_y]

        for para in sample_x:  # generate the initial points
            y = func(para)
            if y < optimal[1]:
                optimal = [para, y]
            sample_y = np.append(sample_y, y)

        print(f'Training with random search')
        sys.stdout.flush()

        for _ in range(max_iter):
            res.nit += 1

            x0 = optimal[0] + np.random.normal(0, .02, para_size)
            prediction = x0

            y = func(prediction)
            res.nfev += 1
            sys.stdout.flush()

            if y < optimal[1]:
                optimal = [prediction, y]
            sample_x = np.append(sample_x, [prediction], axis=0)
            sample_y = np.append(sample_y, y)

        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])

        return res