import sys
sys.path.append("..")

import logging
from dag import Experiment, Recipe
import dill
import os
import utils

logger = logging.getLogger("main")
utils.setup_logging(debug=True)

directory = "../output/08-06-19_seed4"
experiment = Experiment(directory=directory)

# this materializes immediately
x = experiment.spawn_new_tree(
    dataset_name="mnist",
    model_name="models.LeNet",
    init_schema="",
    optimizer_name="sgd",
    seed=4,
)

x = Recipe(
    train={"n_epochs": 30}
)(x)

for _ in range(20):
    pruned = Recipe(
    	prune_schema="../schemas/pruning_schema_lenet_mixedl1.py",
    )(x)
    # x = Recipe(
    #     reinit_schema="../schemas/reinit_schema_lt_lenet.py",
    #     train={"n_epochs": 30},
    # )(pruned) # LT

    x = Recipe(
        train={"n_epochs": 30},
    )(pruned) # finetuning

experiment.run()