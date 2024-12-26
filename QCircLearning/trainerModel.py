import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from .back_minimizer import BackMinimizer
from scipy.optimize import minimize, OptimizeResult
from typing import List, Callable
from .utils import data_augmentation, EarlyStopping, reinitialize_network


class TrainerModel(nn.Module):
    def __init__(self, layers: List[nn.Module] = None, name: str = None):
        super().__init__()
        self.model = nn.Sequential(*layers) if layers else nn.Sequential()
        self.input_shape = layers[0].in_features if layers else None
        self.name = name or "TrainerModel"
        self.loss_fn = nn.MSELoss()

    def forward(self, x, y=None):
        pred = self.model(x)
        if y is not None:
            loss = self.loss_fn(pred, y)
            return loss
        else:
            return pred

    def __str__(self):
        return f"TrainerModel(name={self.name}):\n{self.model}"

    def __repr__(self):
        return self.__str__()

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
                nn.Linear(10, 1),
            ],
            name="default_model",
        )


def model_train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = model(batch_x, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_y.size(0)

    total_loss /= len(data_loader.dataset)
    return total_loss


def NN_opt(func, x0, callback=None, **kwargs):
    para_size = len(x0)
    res = OptimizeResult(nfev=0, nit=0)

    # Default values
    init_data = kwargs.get(
        "init_data", [np.random.uniform(-10, 10, para_size) for _ in range(60)]
    )
    max_iter = kwargs.get("max_iter", 20)
    classical_epochs = kwargs.get("classical_epochs", 20)
    batch_size = kwargs.get("batch_size", 16)
    verbose = kwargs.get("verbose", 0)
    device = kwargs.get("device", "cpu")
    nn_models = kwargs.get(
        "NN_Models",
        [
            TrainerModel.default_model((para_size,)),
        ],
    )
    patience = kwargs.get("patience", 5)
    min_delta = kwargs.get("min_delta", 0.0)

    sample_x = init_data
    sample_y = [func(para) for para in sample_x]
    optimal = [sample_x[np.argmin(sample_y)], np.min(sample_y)]
    if verbose:
        print(f"Training with the neural networks")
    sys.stdout.flush()

    for model in nn_models:
        if verbose:
            print(model)
            sys.stdout.flush()

        for iteration in range(max_iter):
            res.nit += 1
            if verbose:
                print(
                    f"Run ID: {kwargs['run_id']}, Iteration {iteration + 1}/{max_iter}"
                )
                sys.stdout.flush()
            data_loader = DataLoader(
                list(zip(sample_x, sample_y)), batch_size=batch_size, shuffle=True
            )
            reinitialize_network(model)
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-4))
            early_stopping = EarlyStopping(
                patience=patience, min_delta=min_delta, verbose=verbose
            )
            best_model = None
            for epoch in range(classical_epochs):
                total_loss = model_train(model, data_loader, optimizer, device)
                if early_stopping(total_loss):
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1}/{classical_epochs}, Average Loss: {total_loss:.1e}"
                        )
                        sys.stdout.flush()
                    break
                if verbose:
                    print(
                        f"Run ID: {kwargs['run_id']}, Epoch {epoch + 1}/{classical_epochs}, Average Loss: {total_loss:.1e}"
                    )
                    sys.stdout.flush()
                if early_stopping.counter == 0:
                    best_model = copy.deepcopy(model)

            # data augmentation
            model = best_model
            model.eval()
            opt_x = optimal[0]

            backminimizer = BackMinimizer(model)
            new_data_x, new_data_y = data_augmentation(
                opt_x, func, backminimizer, kwargs
            )
            res.nfev += kwargs.get("noise_augment_points", 0) + 1
            # for pred in predictions:
            #     if not np.isfinite(func(pred)):  # Check if `func` can handle the augmented data
            #         print(f"Invalid prediction: {pred}")

            sample_x += new_data_x
            sample_y += new_data_y
            optimal = [sample_x[np.argmin(sample_y)], np.min(sample_y)]

    res.x = np.copy(optimal[0])
    res.fun = np.copy(optimal[1])

    return res


def random_search(func, x0, callback=None, **kwargs):
    para_size = len(x0)
    res = OptimizeResult(nfev=0, nit=0)

    init_data = kwargs.get(
        "init_data", [np.random.uniform(-10, 10, para_size) for _ in range(60)]
    )
    max_iter = kwargs.get("max_iter", 20)
    verbose = kwargs.get("verbose", 0)
    sample_x = init_data
    sample_y = [func(para) for para in sample_x]
    optimal = [sample_x[np.argmin(sample_y)], np.min(sample_y)]
    if verbose:
        print("Training with random search")
    sys.stdout.flush()

    for _ in range(max_iter):
        res.nit += 1

        x0 = optimal[0] + np.random.normal(0, 0.02, para_size)
        y = func(x0)
        res.nfev += 1
        sys.stdout.flush()

        if y < optimal[1]:
            optimal = [x0, y]

        sample_x += [x0]
        sample_y += [y]

    res.x = np.copy(optimal[0])
    res.fun = np.copy(optimal[1])

    return res
