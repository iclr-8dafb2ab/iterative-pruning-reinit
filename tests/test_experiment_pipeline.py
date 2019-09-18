from functools import partial
import numpy as np
import os
import pickle
import pytest
import torch

import conftest
from dag import Experiment, ExperimentState, Recipe
import experiment_pipeline
import initializations
from models import get_model
import pruning as prune

import pathlib
currdir = pathlib.Path(__file__).parent

@pytest.mark.parametrize(
    'model_name, init_schema',
    [
        ("models.LeNet", currdir / "init_schema_lenet.py"),
        ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
    ]
)
def test_spawn_new_tree(dataset_name, model_name, init_schema, optimizer_name, tmp_path):
    # create a new experiment tree and check that the directory is created
    experiment_dir = tmp_path / 'experiment'
    experiment = Experiment(directory=experiment_dir)
    assert experiment_dir.is_dir()

    initial_state = experiment.spawn_new_tree(
        dataset_name=dataset_name, 
        model_name=model_name,
        init_schema=init_schema,
        optimizer_name=optimizer_name,
        seed=1,
    ).get()

    # ensure that the state was created and saved to disk
    assert experiment.root.path
    assert experiment.root.path.parent == experiment_dir
    assert experiment.root.path.exists()

    # check that reloading works and that the loaded model matches the
    # initial model layer by layer
    loaded = ExperimentState.load(experiment.root.path)

    for layer, params in initial_state.model.state_dict().items():
        assert (params == loaded.model.state_dict()[layer]).all()

    # ugly stuff to see if optimizers match.
    # this is mostly due to the optimizer's state_dict saving the IDs of the
    # parameters it acts on, and these not remaining identical over a copy
    # of the experiment.
    for key, item in initial_state.optimizer.state_dict().items():
        if key == "param_groups":
            for id_, param_group in enumerate(item):
                for param_name, param_val in param_group.items():
                    if param_name == "params":
                        assert len(param_val) == len(
                            loaded.optimizer.state_dict()["param_groups"][
                                id_
                            ]["params"]
                        )
                    else:
                        assert (
                            param_val
                            == loaded.optimizer.state_dict()["param_groups"][
                                id_
                            ][param_name]
                        )
        else:
            assert item == loaded.optimizer.state_dict()[key]


@pytest.mark.parametrize(
    'model_name, init_schema',
    [
        ("models.LeNet", currdir / "init_schema_lenet.py"),
        ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
    ]
)
def test_reload_state_by_recipe(dataset_name, model_name, init_schema, optimizer_name, tmp_path):
    # create a new experiment tree and check that the directory is created
    experiment_dir = tmp_path / 'experiment'
    experiment = Experiment(directory=experiment_dir)
    assert experiment_dir.is_dir()

    initial_state = experiment.spawn_new_tree(
        dataset_name=dataset_name, 
        model_name=model_name,
        init_schema=init_schema,
        optimizer_name=optimizer_name,
        seed=0,
    ).get()

    # ensure that the state was created and saved to disk
    assert experiment.root.path
    assert experiment.root.path.parent == experiment_dir
    assert experiment.root.path.exists()

    # reload by specifying same recipe
    loaded_experiment = Experiment(directory=experiment_dir)
    loaded_state = loaded_experiment.spawn_new_tree(
        dataset_name=dataset_name, 
        model_name=model_name,
        init_schema=init_schema,
        optimizer_name=optimizer_name,
        seed=0,
    ).get()

    assert initial_state.filename() == loaded_state.filename()
    assert loaded_state.from_cache

@pytest.mark.parametrize(
    'model_name, init_schema',
    [
        ("models.LeNet", currdir / "init_schema_lenet.py"),
        # ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
    ]
)
def test_no_double_spawn(dataset_name, model_name, init_schema, optimizer_name, tmp_path):
    """Test that a tree (Experiment) cannot have two roots.
    """
    experiment_dir = tmp_path / 'experiment'
    experiment = Experiment(directory=experiment_dir)

    initial_state = experiment.spawn_new_tree(
        dataset_name=dataset_name, 
        model_name=model_name,
        init_schema=init_schema,
        optimizer_name=optimizer_name,
        seed=0,
    ).get()

    # test that you can't add a new root to the same folder
    with pytest.raises(RuntimeError):
        second_initial_state = experiment.spawn_new_tree(
            dataset_name=dataset_name, 
            model_name=model_name,
            init_schema=init_schema,
            optimizer_name=optimizer_name,
            seed=1,
        ).get()

@pytest.mark.parametrize(
    'model_name, init_schema',
    [
        ("models.LeNet", currdir / "init_schema_lenet.py"),
        # ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
    ]
)
def test_seed_in_spawn(dataset_name, model_name, init_schema, optimizer_name, tmp_path):
    """Test that two different random seeds actually produce two different
    ExperimentStates and that these can be differentiated using their
    unique identifier contained in their `filename`. This implies that the
    seed is being correctly taken into consideration when producing the hash.
    """
    experiment_dir = tmp_path / 'experiment'
    experiment = Experiment(directory=experiment_dir)

    initial_state = experiment.spawn_new_tree(
        dataset_name=dataset_name, 
        model_name=model_name,
        init_schema=init_schema,
        optimizer_name=optimizer_name,
        seed=0,
    ).get()

    experiment_dir2 = tmp_path / 'experiment22'
    experiment2 = Experiment(directory=experiment_dir2)

    initial_state2 = experiment2.spawn_new_tree(
        dataset_name=dataset_name, 
        model_name=model_name,
        init_schema=init_schema,
        optimizer_name=optimizer_name,
        seed=1,
    ).get()

    assert initial_state.filename() != initial_state2.filename()


class TestRecipe:
    @pytest.mark.parametrize(
        'model_name, schema',
        [
            ("models.LeNet", currdir / "pruning_schema_lenet.py"),
            ("torchvision.models.AlexNet", currdir / "pruning_schema_alexnet.py")
        ]
    )
    def test_prune(self, model_name, schema):
        model = get_model(model_name)
        schema = eval(open(schema).read())
        modules = np.unique([k for k, _ in schema.keys()])
        experiment_pipeline.prune_model(model, schema, seed=0)

        for module_name in modules:
            module = dict(model.named_modules())[module_name]
            assert "weight_orig" in dict(module.named_parameters())
            assert "bias_orig" in dict(module.named_parameters())
            assert "weight_mask" in dict(module.named_buffers())
            assert "bias_mask" in dict(module.named_buffers())
            # TODO: test that there are hooks


    @pytest.mark.parametrize(
        'model_name, schema',
        [
            ("models.LeNet", currdir / "init_schema_lenet.py"),
            ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
        ]
    )
    def test_init(self, model_name, schema):
        model = get_model(model_name)
        schema = eval(open(schema).read())

        experiment_pipeline.init_model(model, schema, seed=123)
        parameters = np.unique(
            [".".join((k1, k2)) for k1, k2 in schema.keys()]
        )

        for param_name in parameters:
            param = dict(model.named_parameters())[param_name]
            if "weight" in param_name:
                assert torch.all(param.data[1:] == 0.5)
                assert torch.all(param.data[0] == -2.5)
            if "bias" in param_name:
                assert torch.all(param.data[1:] == 2)
                assert torch.all(param.data[0] == -1)

                
    @pytest.mark.parametrize(
        'model_name, init_schema',
        [
            ("models.LeNet", currdir / "init_schema_lenet.py"),
            # ("torchvision.models.AlexNet", currdir / "init_schema_alexnet.py")
        ]
    )           
    def test_recipe_seed(self, dataset_name, model_name, init_schema, optimizer_name, tmp_path):
        experiment_dir = tmp_path / 'experiment'
        experiment = Experiment(directory=experiment_dir)

        initial_state = experiment.spawn_new_tree(
            dataset_name=dataset_name, 
            model_name=model_name,
            init_schema=init_schema,
            optimizer_name=optimizer_name,
            seed=999,
        )
        assert initial_state.get().seed == 999

        r = Recipe(train={"n_epochs": 1})
        assert r.seed is None

        new_state = r(initial_state)
        assert r.seed == initial_state.get().seed
