import sys, os
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)

import pytest
import torch
from functools import partial
import pruning as prune

# Fixtures
@pytest.fixture(params=["MNIST", "Mnist", "cifar-10", "CIFAR-100"])
def dataset_name(request):
    return request.param

@pytest.fixture(params=[1, 3])
def num_channels(request):
    return request.param

@pytest.fixture(params=[torch.device("cpu"), torch.device("cuda")])
def device(request):
    return request.param


@pytest.fixture(params=["models.LeNet", "torchvision.models.AlexNet"])
def model_name(request):
    return request.param


@pytest.fixture(params=["adam", "Adam", "SGD"])
def optimizer_name(request):
    return request.param
