import logging

import torch

from dag import Experiment, Recipe
from utils import setup_logging

setup_logging(debug=True)

experiment = Experiment(directory="./end-to-end-experiment")

# This materializes immediately
initial_state = experiment.spawn_new_tree(
    dataset_name="mnist",
    model_name="models.LeNet",
    init_schema="schemas/init_kaiminguniform_lenet.py",
    optimizer_name="sgd",
    seed=0,
    device=torch.device("cpu"),
)

# These are computed lazily until .run() is called. A Recipe represents a
# way to transition from one state (read: model) to the following via
# pruning, finetuning, and reinitializing
state1 = Recipe(
    train={"n_epochs": 1},
    prune_schema="schemas/pruning_schema_lenet_unstructuredl1.py",
)(initial_state)

state2 = Recipe(reinit_schema="schemas/reinit_schema_lenet.py")(state1)

state3 = Recipe(
    train={"n_epochs": 1},
    prune_schema="schemas/pruning_schema_lenet_unstructuredl1.py",
    reinit_schema="schemas/reinit_schema_lenet.py",
)(state2)

final_state = Recipe(train={"n_epochs": 1})(state3)

# When .run() is called, we execute the full experiment graph
experiment.run()
