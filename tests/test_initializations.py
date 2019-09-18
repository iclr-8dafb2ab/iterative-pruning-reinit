from functools import partial
import os 
import pytest
import torch
from torchvision.models import AlexNet
import numpy as np

import conftest
from dag import Experiment, ExperimentState
from experiment_pipeline import prune_model
import initializations
from initializations import (
    init_copy_,
    get_matching_parameter,
    rewind_initialization,
    zhou_initialization
)
import pruning as prune
import pdb

import pathlib
currdir = pathlib.Path(__file__).parent

class TestInit:
    def test_init_copy(self):
        t = torch.randn(2, 3)
        t_other = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(dtype=t.dtype)
        init_copy_(t, t_other)
        assert torch.equal(t, t_other)

    def test_init_copy_error(self):
        t = torch.randn(2, 3)
        t_other = torch.tensor([[1, 2], [4, 5]]).to(dtype=t.dtype)
        with pytest.raises(ValueError):
            init_copy_(t, t_other)

    def test_get_matching_parameter(self):
        # pick a model
        model = AlexNet()
        # define how to prune it
        schema = eval(open(currdir / "pruning_schema_alexnet.py").read())
        # prune the model according to the schema
        prune_model(model, schema, seed=0)

        correct = model.features[0].weight_orig
        retrieved = get_matching_parameter("features.0.weight", model)
        assert torch.equal(correct, retrieved)

        correct = model.features[0].weight_orig
        retrieved = get_matching_parameter("features.0.weight_orig", model)
        assert torch.equal(correct, retrieved)

        correct = model.features[3].bias
        retrieved = get_matching_parameter("features.3.bias_orig", model)
        assert torch.equal(correct, retrieved)

        with pytest.raises(KeyError):
            get_matching_parameter("blah", model)

    @pytest.mark.parametrize(
        'model_name, init_schema',
        [
            ("models.LeNet", currdir / "init_schema_lenet.py"),
            ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
        ]
    )
    def test_rewind_initialization(
        self, dataset_name, model_name, init_schema, optimizer_name, tmp_path
    ):
        """Tests that a rewounded model matches the one found in root_path.
        Gets the first parameter of the model, modifies it manually, and then
        tries to rewind it.
        """
        experiment_dir = tmp_path / 'experiment'
        experiment = Experiment(directory=experiment_dir)

        # create a new experiment state
        initial_state = experiment.spawn_new_tree(
            dataset_name=dataset_name, 
            model_name=model_name,
            init_schema=init_schema,
            optimizer_name=optimizer_name,
            seed=3,
        ).get()

        # get first parameter of the model at initializatiion
        param_name, param = next(initial_state.model.named_parameters())
        init_param = param.clone()

        # reset the parameter manually and check it's different
        _, param = next(initial_state.model.named_parameters())
        torch.nn.init.constant_(param, val=3.0)
        assert not torch.equal(init_param, param)

        # rewind initialization and check the param went back to its original
        # state
        initial_state.experiment_object.root = initial_state  # mock
        rewind_initialization(param, param_name, initial_state)
        _, rewound_param = next(initial_state.model.named_parameters())
        assert torch.equal(init_param, rewound_param)
    

    @pytest.mark.parametrize(
        'model_name, init_schema',
        [
            ("models.LeNet", currdir / "init_schema_lenet.py"),
            ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
        ]
    )
    def test_zhou_initialization(
        self, dataset_name, model_name, init_schema, optimizer_name, tmp_path
    ):
        """Tests zhou initialization of model
        """
        experiment_dir = tmp_path / 'experiment'
        experiment = Experiment(directory=experiment_dir)

        # create a new experiment state
        initial_state = experiment.spawn_new_tree(
            dataset_name=dataset_name, 
            model_name=model_name,
            init_schema=init_schema,
            optimizer_name=optimizer_name,
            seed=2,
        ).get()

        init_schema = eval(open(init_schema).read())

        parameter_names = np.unique([".".join((k1, k2)) for k1, k2 in init_schema.keys()])
        params = [get_matching_parameter(param_name, initial_state.model) for param_name in parameter_names]
        const_sign = [param.std().item() * torch.sign(param.data) for param in params]
        
        initial_state.experiment_object.root = initial_state  # mock

        for param_name, param, const_layer in zip(parameter_names, params, const_sign):
            zhou_initialization(param, param_name, initial_state)
            assert torch.equal(
                const_layer,
                get_matching_parameter(param_name, initial_state.model)
            )
