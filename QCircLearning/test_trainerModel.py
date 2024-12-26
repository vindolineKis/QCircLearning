import os
import copy
import random
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from QCircLearning.trainerModel import model_train, TrainerModel
import torch.optim as optim
from .circuit_struct import VCircuitConstructor
from .evaluate import Evaluator
from .utils import data_augmentation, reinitialize_network
from .back_minimizer import BackMinimizer

# 设置随机种子
SEED = 45
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# 确定性操作
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.fixture
def toy_network():
    return TrainerModel(
        layers=[
            nn.Linear(5, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 18),
            nn.ReLU(),
            nn.Linear(18, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        ]
    ).double()


@pytest.fixture
def setup(toy_network):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = toy_network.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 1
    num_point = 10
    epoch = 1000
    return model, optimizer, device, batch_size, num_point, epoch


def test_model_train_full_random(setup):
    model, optimizer, device, batch_size, num_point, epoch = setup

    sample_x = [np.random.uniform(-2 * np.pi, 2 * np.pi, 5) for _ in range(num_point)]
    sample_y = [np.random.uniform(-2 * np.pi, 2 * np.pi, 1) for _ in range(num_point)]
    data_loader = DataLoader(
        list(zip(sample_x, sample_y)), batch_size=batch_size, shuffle=True
    )
    best_loss = float("inf")
    for i in range(epoch):
        total_loss = model_train(model, data_loader, optimizer, device)
        print("Epoch: ", i, "Loss: ", total_loss)
        if total_loss < best_loss:
            best_loss = total_loss

    print(best_loss)
    assert best_loss < 1e-10, "Loss should be small enough"


def test_model_train_with_circ(setup):
    model, optimizer, device, batch_size, num_point, epoch = setup
    vqc = VCircuitConstructor.get_vqc(1, "one_layer")
    target_qc = vqc["circuit"].assign_parameters([np.pi, np.pi / 2, 0, 0, 0])

    # Define cost function
    evaluator = Evaluator("para", target=target_qc, vqc=vqc["circuit"])
    cost = lambda x: 1 - evaluator.evaluate(x)

    sample_x = [np.random.uniform(-2 * np.pi, 2 * np.pi, 5) for _ in range(num_point)]
    sample_y = [
        np.array(cost(x)).reshape(
            1,
        )
        for x in sample_x
    ]
    data_loader = DataLoader(
        list(zip(sample_x, sample_y)), batch_size=batch_size, shuffle=True
    )
    best_loss = float("inf")
    for i in range(epoch):
        total_loss = model_train(model, data_loader, optimizer, device)
        print("Epoch: ", i, "Loss: ", total_loss)
        if total_loss < best_loss:
            best_loss = total_loss

    print(best_loss)
    assert best_loss < 1e-10, "Loss should be small enough"


def test_model_train_with_circ_augment(setup):
    model, optimizer, device, batch_size, num_point, epoch = setup

    epoch = 1000
    vqc = VCircuitConstructor.get_vqc(1, "one_layer")
    target_qc = vqc["circuit"].assign_parameters([np.pi, np.pi / 2, 0, 0, 0])

    # Define cost function
    evaluator = Evaluator("para", target=target_qc, vqc=vqc["circuit"])
    cost = lambda x: 1 - evaluator.evaluate(x)

    # generate test data
    sample_x = [np.random.uniform(-2 * np.pi, 2 * np.pi, 5) for _ in range(num_point)]
    sample_y = [cost(x) for x in sample_x]

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    reinitialize_network(model)
    ref_model = copy.deepcopy(model)

    for it in range(5):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        reinitialize_network(model)
        for a, b in zip(list(model.parameters()), list(ref_model.parameters())):
            assert torch.allclose(a, b), "reinitialization wrong"

        data_loader = DataLoader(
            list(zip(sample_x, sample_y)), batch_size=batch_size, shuffle=True
        )

        best_loss = float("inf")
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for i in range(epoch):
            total_loss = model_train(model, data_loader, optimizer, device)
            print(f"Iter: {it+1} Epoch: {i} Loss: {total_loss:.1e}")
            if total_loss < best_loss:
                best_loss = total_loss

        assert best_loss < 1e-8, "Loss should be small enough"

        backminimizer = BackMinimizer(model)

        new_data_x, new_data_y = data_augmentation(
            sample_x[np.argmin(sample_y)],
            cost,
            backminimizer,
            {
                "noise_augment_points": 0,
                "periodic_augment_points": 1,
            },
        )
        sample_x += new_data_x
        sample_y += new_data_y
