import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import List, Callable


class TrainerModel(nn.Module):
    def __init__(self, layers: List[nn.Module] = None, name: str = None):
        super(TrainerModel, self).__init__()
        self.model = nn.Sequential(*layers) if layers else nn.Sequential()
        self.name = name or "TrainerModel"
        self.double()  # Use double precision for consistency

    def forward(self, x):
        return self.model(x)

    def back_minimize(
        self, x0: np.ndarray = None, method: str = "L-BFGS-B", verbose: int = 0
    ):
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

        result = minimize(
            to_minimize_with_grad,
            x,
            bounds=[(-np.pi * 2, np.pi * 2)] * len(x),
            jac=True,
            method=method,
            tol=1e-6,
            options={
                "disp": None,
                "maxls": 20,
                "iprint": -1,
                "eps": 1e-7,
                "ftol": 1e-6,
                "maxiter": 1500,
                "maxcor": 12,
                "maxfun": 1500,
            },
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

    @staticmethod
    def simple_model(input_shape: tuple):
        return TrainerModel(
            layers=[
                nn.Linear(input_shape[0], 32),
                nn.ELU(),
                nn.Linear(32, 8),
                nn.Sigmoid(),
                nn.Linear(8, 1),
            ],
            name="simple_model",
        )

    @staticmethod
    def NN_opt(func, x0, callback=None, **kwargs):
        para_size = len(x0)
        res = OptimizeResult(nfev=0, nit=0)
        res.nfev = 0
        res.nit = 0

        # Default values
        init_data = kwargs.get("init_data", np.random.uniform(-10, 10, (60, para_size)))
        max_iter = kwargs.get("max_iter", 20)
        classical_epochs = kwargs.get("classical_epochs", 20)
        batch_size = kwargs.get("batch_size", 16)
        verbose = kwargs.get("verbose", 0)
        nn_models = kwargs.get(
            "NN_Models",
            [
                TrainerModel.default_model((para_size,)),
                TrainerModel.simple_model((para_size,)),
            ],
        )
        patience = kwargs.get("patience", 100)

        sample_y, sample_x = np.array([]), np.empty((0, para_size))
        optimal = [None, float("inf")]

        # Generate initial sample data
        for para in init_data:
            y = func(para)
            if y < optimal[1]:
                optimal = [para, y]
            sample_x = np.vstack([sample_x, para])
            sample_y = np.append(sample_y, y)

        print("Training with the neural networks")
        sys.stdout.flush()

        for model in nn_models:
            print(model)
            sys.stdout.flush()

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=patience,
                verbose=True,
                min_lr=1e-6,
            )
            precision_threshold = 1e-8

            for iteration in range(max_iter):
                res.nit += 1
                print(f"Iteration {iteration + 1}/{max_iter}")

                model.train()
                for epoch in range(classical_epochs):
                    total_loss = 0.0
                    permutation = torch.randperm(sample_x.shape[0])

                    for i in range(0, sample_x.shape[0], batch_size):
                        indices = permutation[i : i + batch_size]
                        batch_x, batch_y = sample_x[indices], sample_y[indices]

                        inputs = torch.tensor(batch_x, dtype=torch.float64)
                        targets = torch.tensor(batch_y, dtype=torch.float64).unsqueeze(
                            -1
                        )

                        optimizer.zero_grad()
                        outputs = model(inputs)

                        loss = criterion(outputs, targets)
                        total_loss += loss.item()

                        loss.backward()
                        # print the gradients
                        # print(model.model[0].weight.grad)
                        optimizer.step()

                    if verbose:
                        avg_loss = total_loss / (sample_x.shape[0] // batch_size)
                        print(
                            f"Epoch {epoch + 1}/{classical_epochs}, Average Loss: {avg_loss:.1e}"
                        )
                        sys.stdout.flush()
                scheduler.step(total_loss)
                current_lr = optimizer.param_groups[0]["lr"]
                if current_lr < precision_threshold:
                    print(
                        f"Training stopped as learning rate reached precision threshold: {current_lr:.1e}"
                    )
                    break
                # Prediction and updating optimal parameters
                x0 = optimal[0]
                prediction0 = model.back_minimize(
                    x0=x0, method="L-BFGS-B", verbose=verbose
                )
                y0 = func(prediction0)
                res.nfev += 1

                if y0 < optimal[1]:
                    optimal = [prediction0, y0]

                predictions = np.vstack(
                    [prediction0, prediction0 + np.pi * 2, prediction0 - np.pi * 2]
                )
                sample_x = np.concatenate([sample_x, predictions], axis=0)
                sample_y = np.concatenate([sample_y, [y0] * predictions.shape[0]])

        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])

        return res

    @staticmethod
    def random_search(func, x0, callback=None, **kwargs):
        para_size = len(x0)
        res = OptimizeResult(nfev=0, nit=0)

        init_data = kwargs.get("init_data", np.random.uniform(-10, 10, (60, para_size)))
        max_iter = kwargs.get("max_iter", 20)

        sample_x = init_data
        sample_y = np.array([func(para) for para in sample_x])
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

            sample_x = np.vstack([sample_x, x0])
            sample_y = np.append(sample_y, y)

        res.x = np.copy(optimal[0])
        res.fun = np.copy(optimal[1])

        return res
