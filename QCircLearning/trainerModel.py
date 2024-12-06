import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import List, Callable
from .utils import data_augment


class TrainerModel(nn.Module):
    def __init__(self, layers: List[nn.Module] = None, name: str = None):
        super().__init__()
        self.model = nn.Sequential(*layers) if layers else nn.Sequential()
        self.name = name or "TrainerModel"
        self.loss_fn = nn.MSELoss()

    def forward(self, x, y=None):
        pred = self.model(x)
        if y is not None:
            loss = self.loss_fn(pred, y)
            return loss
        else:
            return pred

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


def back_minimize(
    model, x0: np.ndarray = None, method: str = "L-BFGS-B", verbose: int = 0, **kwargs
):
    x_list, f_list, gradient_list = [], [], []

    def to_minimize(x):
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        return model(x_tensor).detach().numpy()

    x = x0 if x0 is not None else np.random.rand(self.model[0].in_features)

    def to_minimize_with_grad(x):
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        x_list.append(x)
        loss = model(x_tensor)
        f_list.append(loss.item())
        loss.backward()
        gradients = x_tensor.grad.numpy()
        gradient_list.append(gradients)
        return loss.item(), gradients

    result = minimize(
        to_minimize_with_grad,
        x,
        bounds=[(-np.pi * 2, np.pi * 2)] * len(x),
        jac=True,
        method=method,
        tol=1e-6,
        options=kwargs["back_minimize_options"],
    )

    if verbose:
        print("Optimization result:", result)
        print(f"Optimization converged: {result.success}")
        print(f"Stored x_list: {x_list[-1]}")
        print(f"Stored f_list: {f_list}")
        print(f"Stored gradient_list: {gradient_list[-1]}")
        if result.success:
            print("Optimization converged successfully.")
        else:
            print("Optimization did not converge. Reason:", result.message)

    return result.x


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
    sample_x = init_data
    sample_y = [func(para) for para in sample_x]
    optimal = [sample_x[np.argmin(sample_y)], np.min(sample_y)]

    if verbose:
        print("Training with the neural networks")
        sys.stdout.flush()

    for model in nn_models:
        if verbose:
            print(model)
            sys.stdout.flush()

        optimizer = optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-4))

        for iteration in range(max_iter):
            res.nit += 1
            if verbose:
                print(f"Run ID: {kwargs['run_id']}, Iteration {iteration + 1}/{max_iter}")

            data_loader = DataLoader(
                list(zip(sample_x, sample_y)), batch_size=batch_size, shuffle=True
            )
            model.train()
            for epoch in range(classical_epochs):
                total_loss = 0.0

                for batch_x, batch_y in data_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = model(batch_x, batch_y)
                    loss.backward()
                    # print the gradients
                    # print(model.model[0].weight.grad)
                    optimizer.step()
                    total_loss += loss.item() * batch_y.size(0)

                total_loss /= len(data_loader.dataset)
                if verbose:
                    print(
                        f"Run ID: {kwargs['run_id']}, Epoch {epoch + 1}/{classical_epochs}, Average Loss: {total_loss:.1e}"
                    )
                    sys.stdout.flush()

            # Prediction and updating optimal parameters
            x0 = optimal[0]
            prediction0 = back_minimize(model, x0=x0, method="L-BFGS-B", **kwargs)
            y0 = func(prediction0)
            res.nfev += 1

            optimal = [prediction0, y0] if y0 < optimal[1] else optimal

            augment_points = kwargs.get("augment_points", 2)
            predictions = data_augment(prediction0, augment_points, shift=np.pi * 2)
            sample_x += predictions
            sample_y += [y0] * len(predictions)

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

    sample_x = init_data
    sample_y = [func(para) for para in sample_x]
    optimal = [sample_x[np.argmin(sample_y)], np.min(sample_y)]

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
