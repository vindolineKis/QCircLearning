import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from QCircLearning.trainerModel import model_train, TrainerModel
import torch.optim as optim
from .circuit_struct import VCircuitConstructor
from .evaluate import Evaluator


@pytest.fixture
def toy_network():
    torch.manual_seed(42)
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
    assert best_loss < 1e-10, "Loss should be small enough"

    print(best_loss)


def test_model_train_with_circ(setup):
    model, optimizer, device, batch_size, num_point, epoch = setup
    vqc = VCircuitConstructor.get_vqc(1, "one_layer")
    target_qc = vqc["circuit"].assign_parameters([np.pi, np.pi / 2, 0, 0, 0])

    # Define cost function
    evaluator = Evaluator("para", target=target_qc, vqc=vqc["circuit"])
    cost = lambda x: 1 - evaluator.evaluate(x)

    sample_x = [np.random.uniform(-2 * np.pi, 2 * np.pi, 5) for _ in range(num_point)]
    sample_y = [np.array(cost(x)).reshape(1,) for x in sample_x]
    data_loader = DataLoader(
        list(zip(sample_x, sample_y)), batch_size=batch_size, shuffle=True
    )
    best_loss = float("inf")
    for i in range(epoch):
        total_loss = model_train(model, data_loader, optimizer, device)
        print("Epoch: ", i, "Loss: ", total_loss)
        if total_loss < best_loss:
            best_loss = total_loss
    assert best_loss < 1e-10, "Loss should be small enough"

    print(best_loss)
