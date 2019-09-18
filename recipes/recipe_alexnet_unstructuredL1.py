import sys
sys.path.append("..")

import logging
from dag import Experiment, Recipe
import dill
import os
import utils

logger = logging.getLogger("main")
utils.setup_logging(debug=True)

directory = "../output/alexnet_seed4"
experiment = Experiment(directory=directory)

# this materializes immediately
x = experiment.spawn_new_tree(
    dataset_name="mnist",
    model_name="torchvision.models.AlexNet",
    init_schema="",
    optimizer_name="sgd",
    seed=4,
)

x = Recipe(
    train={"n_epochs": 30}
)(x)

for _ in range(20):
    # finetune
    pruned = Recipe(
        prune_schema="../schemas/pruning_schema_alexnet_unstructuredl1.py",
    )(x)
    x = Recipe(
        reinit_schema="../schemas/reinit_schema_lt_alexnet.py",
        train={"n_epochs": 30},
    )(pruned) # LT

    # x = Recipe(
    #     train={"n_epochs": 20},
    # )(pruned) # finetuning

experiment.run()