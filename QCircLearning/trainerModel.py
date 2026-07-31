import sys
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from .back_minimizer import BackMinimizer
from scipy.optimize import minimize, OptimizeResult
from typing import List, Callable
from .utils import data_augmentation, EarlyStopping, reinitialize_network
import time
from cmaes import CMA


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
            loss = self.loss_fn(pred, y.unsqueeze(-1))
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
    
    model.to(device)
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

    # logging
    logger = logging.getLogger(__name__)
    adapter = logging.LoggerAdapter(logger, {"run_id": kwargs["run_id"]})

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
    # # embeding_data(sample_x, kwargs)
    # sample_x = embeding_data(sample_x, kwargs)

    for model in nn_models:
        if verbose:
            print(model)
            sys.stdout.flush()

        early_stop_epoch = []
        
        for iteration in range(max_iter):
            # res.nit += 1
            if verbose:
                print(
                    f"Run ID: {kwargs['run_id']}, Iteration {iteration + 1}/{max_iter}"
                )
                sys.stdout.flush()
            data_loader = DataLoader(
                list(zip(sample_x, sample_y)), batch_size=batch_size, shuffle=True
            )
            if kwargs.get("reinitialize_model", False):
                reinitialize_network(model)
                track=reinitialize_network(model)
                if verbose:
                    print(f"Run ID: {kwargs['run_id']}, Model reinitialized:{track}")  


            model.train()
            optimizer = optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-4))
            if kwargs.get("use_scheduler", False):
                scheduler_kwargs = kwargs.get("scheduler_kwargs", {})
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **scheduler_kwargs)
                
                # if verbose:
                #     print(f"mode in scheduler: {scheduler.mode}")
                #     print(f"factor in scheduler: {scheduler.factor}")
                #     print(f"Initial lr: {optimizer.param_groups[0]['lr']}")
                #     print(f"patience in scheduler: {scheduler.patience}")
                #     sys.stdout.flush()

            early_stopping = EarlyStopping(
                patience=patience, min_delta=min_delta, verbose=verbose
            )
            best_model_state = None

            for epoch in range(classical_epochs):
                # record the time cost for each epoch
                # start_time_epoch = time.time()
                total_loss = model_train(model, data_loader, optimizer, device)
                if kwargs.get("use_scheduler", False):
                    scheduler.step(total_loss)
                if early_stopping(total_loss):
                    message = f"Iter {iteration+1}/{max_iter}, Early stopping at epoch {epoch + 1}/{classical_epochs}, Best Loss: {early_stopping.best_loss:.1e}"
                    if verbose:
                        print(f"Run id {kwargs['run_id']}, {message}")
                        sys.stdout.flush()
                    adapter.info(message)
                    early_stop_epoch.append(epoch)
                    break
                # if verbose:
                #     print(f"current lr: {optimizer.param_groups[0]['lr']}")
                #     print(
                #         f"Run ID: {kwargs['run_id']}, Epoch {epoch + 1}/{classical_epochs}, Average Loss: {total_loss:.1e}"
                #     )
                #     sys.stdout.flush()


                # TODO: test deepcopy time
            
                # start_time_deepcopy = time.time()
                best_model_state = copy.deepcopy(model.state_dict()) if early_stopping.reset else best_model_state
                # best_model_state = model.state_dict() if early_stopping.reset else best_model_state
                # if verbose:
                #     print(f"Deepcopy time: {time.time() - start_time_deepcopy}")
                #     print(f"Time cost of each epoch: {time.time() - start_time_epoch}")
                #     sys.stdout.flush()
                # record the time cost of each epoch

                
        
            # model.load_state_dict(best_model_state)
            model.eval()
            opt_x = optimal[0]+np.random.normal(0, 0.02, para_size)

            backminimizer = BackMinimizer(model)

            # data augmentation
            new_data_x, new_data_y = data_augmentation(
                opt_x, func, backminimizer, kwargs
            )
            # res.nfev += kwargs.get("noise_augment_points", 0) + 1
            # for pred in predictions:
            #     if not np.isfinite(func(pred)):  # Check if `func` can handle the augmented data
            #         print(f"Invalid prediction: {pred}")

            sample_x += new_data_x
            sample_y += new_data_y
            optimal = [sample_x[np.argmin(sample_y)], np.min(sample_y)]

        adapter.info(f"Average early stopping epoch: {np.mean(early_stop_epoch)}")

    res.x = np.copy(optimal[0])
    res.fun = np.copy(optimal[1])

    return res


def random_search(func, x0, callback=None, **kwargs):
    para_size = len(x0)
    res = OptimizeResult(nfev=0, nit=0)

    init_data = kwargs.get(
        "init_data", [np.random.uniform(-10, 10, para_size) for _ in range(60)]
    )

    max_iter = kwargs.get("max_iter", kwargs.get("maxiter", 20))
    verbose = kwargs.get("verbose", 0)

    sample_x = list(copy.deepcopy(init_data))
    sample_y = [func(para) for para in sample_x]

    best_idx = int(np.argmin(sample_y))
    optimal = [np.array(sample_x[best_idx]).copy(), float(sample_y[best_idx])]

    if verbose:
        print("Training with random search")
        sys.stdout.flush()

    for _ in range(max_iter):
        x_new = optimal[0] + np.random.normal(0, 0.02, para_size)
        y_new = func(x_new)

        if y_new < optimal[1]:
            optimal = [x_new.copy(), float(y_new)]

        sample_x.append(x_new.copy())
        sample_y.append(float(y_new))

        if callback is not None:
            callback(x_new)

    res.x = np.copy(optimal[0])
    res.fun = float(optimal[1])
    res.nit = max_iter
    res.nfev = len(sample_y)
    res.success = True
    res.message = "Random search finished."

    return res


def minimize_cmaes(
    func,
    x0,
    callback=None,
    sigma=0.5,
    maxiter=None,
    population_size=None,
    **kwargs
):
    x0 = np.asarray(x0, dtype=float)

    if maxiter is None:
        maxiter = kwargs.get("max_iter", 100)

    if population_size is None:
        optimizer = CMA(mean=x0, sigma=sigma)
    else:
        optimizer = CMA(mean=x0, sigma=sigma, population_size=population_size)

    best_x = x0.copy()
    best_y = func(best_x)
    nfev = 1

    for _ in range(maxiter):
        solutions = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            y = func(x)
            nfev += 1

            solutions.append((x, y))

            if y < best_y:
                best_x = x.copy()
                best_y = float(y)

        optimizer.tell(solutions)

        if callback is not None:
            callback(best_x)

    return OptimizeResult(
        x=best_x.copy(),
        fun=float(best_y),
        nit=maxiter,
        nfev=nfev,
        success=True,
        message="CMA-ES finished."
    )


def minimize_spsa(
    func,
    x0,
    callback=None,
    maxiter=None,
    a=0.1,
    c=0.1,
    A=10.0,
    alpha=0.602,
    gamma=0.101,
    **kwargs
):
    if maxiter is None:
        maxiter = kwargs.get("max_iter", 200)

    x = np.asarray(x0, dtype=float).copy()

    best_x = x.copy()
    best_y = func(x)
    nfev = 1

    for k in range(maxiter):
        ak = a / ((k + 1 + A) ** alpha)
        ck = c / ((k + 1) ** gamma)

        delta = np.random.choice([-1.0, 1.0], size=x.shape)

        x_plus = x + ck * delta
        x_minus = x - ck * delta

        y_plus = func(x_plus)
        y_minus = func(x_minus)
        nfev += 2

        ghat = (y_plus - y_minus) / (2.0 * ck * delta)

        x = x - ak * ghat

        y = func(x)
        nfev += 1

        if y < best_y:
            best_x = x.copy()
            best_y = float(y)

        if callback is not None:
            callback(x)

    return OptimizeResult(
        x=best_x.copy(),
        fun=float(best_y),
        nit=maxiter,
        nfev=nfev,
        success=True,
        message="SPSA finished."
    )

def minimize_smo(
    func,
    x0,
    callback=None,
    maxiter=None,
    tol=1e-8,
    order="cyclic",
    wrap_angles=True,
    reestimate_every=None,
    **kwargs
):
    """
    Single-parameter SMO optimizer for periodic parametrized circuits.

    Assumption:
    With all other parameters fixed, the objective as a function of one
    parameter is approximately of the form

        f(theta_j) = A cos(theta_j) + B sin(theta_j) + C

    which is the standard setting for Pauli-rotation-based variational circuits.

    Parameters
    ----------
    func : callable
        Objective function.
    x0 : array-like
        Initial point.
    callback : callable or None
        callback(xk)
    maxiter : int or None
        Number of sweeps over all parameters.
    tol : float
        Stop if improvement per sweep is below tol.
    order : str
        "cyclic" or "random"
    wrap_angles : bool
        Whether to wrap parameters into [0, 2*pi).
    reestimate_every : int or None
        Re-evaluate current best point every fixed number of parameter updates
        to reduce accumulated fitting error in noisy settings.

    Returns
    -------
    OptimizeResult
    """
    if maxiter is None:
        maxiter = kwargs.get("max_iter", 50)

    x = np.asarray(x0, dtype=float).copy()

    def maybe_wrap(v):
        return np.mod(v, 2 * np.pi) if wrap_angles else v

    x = maybe_wrap(x)
    dim = len(x)

    best_x = x.copy()
    best_y = func(best_x)
    nfev = 1
    nit = 0
    step_counter = 0

    def smo_single_step(x_current, j):
        theta0 = x_current[j]

        x_base = x_current.copy()
        x_plus = x_current.copy()
        x_minus = x_current.copy()

        x_plus[j] = theta0 + np.pi / 2
        x_minus[j] = theta0 - np.pi / 2

        x_plus = maybe_wrap(x_plus)
        x_minus = maybe_wrap(x_minus)

        f0 = func(x_base)
        f_plus = func(x_plus)
        f_minus = func(x_minus)

        # Recover f(theta) = A cos(theta) + B sin(theta) + C
        C = 0.5 * (f_plus + f_minus)
        A = f0 - C
        B = 0.5 * (f_minus - f_plus)

        amp = np.hypot(A, B)

        # Flat direction: skip update
        if amp < 1e-12:
            return x_current.copy(), float(f0), 3

        phi = np.arctan2(B, A)
        theta_star = phi + np.pi  # minimizer of r cos(theta - phi) + C

        x_new = x_current.copy()
        x_new[j] = theta_star
        x_new = maybe_wrap(x_new)

        f_new = func(x_new)

        return x_new, float(f_new), 4

    for sweep in range(maxiter):
        y_before = best_y

        if order == "random":
            indices = np.random.permutation(dim)
        else:
            indices = range(dim)

        for j in indices:
            x_new, y_new, evals = smo_single_step(best_x, j)
            nfev += evals
            step_counter += 1

            if y_new <= best_y:
                best_x = x_new
                best_y = y_new

            if callback is not None:
                callback(best_x)

            if reestimate_every is not None and step_counter % reestimate_every == 0:
                best_y = float(func(best_x))
                nfev += 1

        nit += 1

        if y_before - best_y < tol:
            return OptimizeResult(
                x=best_x.copy(),
                fun=float(best_y),
                nit=nit,
                nfev=nfev,
                success=True,
                message="SMO converged."
            )

    return OptimizeResult(
        x=best_x.copy(),
        fun=float(best_y),
        nit=nit,
        nfev=nfev,
        success=True,
        message="SMO reached maxiter."
    )