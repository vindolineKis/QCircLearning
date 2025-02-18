import numpy as np


def data_augment_periodic(data, n_points, shift):
    if n_points < 0:
        raise ValueError("n_points should be greater than 0")
    new_data = [data]
    for _ in range(n_points):
        new_data.append(data + np.random.choice([-1, 0, 1], len(data)) * shift)
    return new_data


def data_augment_noise(data, noise_strength):
    new_data = data + np.random.randn(*np.shape(data)) * noise_strength
    return new_data


def data_augmentation(data, circ_evaluate, backminimizer, config):
    noise_augment_points = config.get("noise_augment_points", 2)
    if noise_augment_points < 0:
        raise ValueError("noise_augment_points should be greater than 0")
    periodic_augment_points = config.get("periodic_augment_points", 2)

    refine_x = backminimizer.back_minimize(
        x0=data, method="L-BFGS-B", **config
    )

    new_data_x = []
    new_data_y = []

    for index in range(noise_augment_points + 1):
        noise_strengh = config.get("noise_strengh", 0.01)
        circ_data_x = data_augment_noise(refine_x, noise_strengh) if index != 0 else refine_x
        circ_output = circ_evaluate(circ_data_x)

        periodic_x = data_augment_periodic(
            circ_data_x, periodic_augment_points, shift=np.pi * 2
        )
        periodic_y = [circ_output] * len(periodic_x)
        new_data_x += periodic_x
        new_data_y += periodic_y

    return new_data_x, new_data_y


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        # self.early_stop = False

    @property
    def reset(self):
        return self.counter == 0

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        return False


def reinitialize_network(model):
    """
    Re-initializes the parameters of a PyTorch network in place
    (without constructing a new model).

    model (nn.Module): The model to re-init.
    """
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
