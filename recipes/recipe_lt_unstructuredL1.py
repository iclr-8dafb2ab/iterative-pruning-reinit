import sys
sys.path.append("..")

import logging
from dag import Experiment, Recipe
import dill
import os
import utils

logger = logging.getLogger("main")
utils.setup_logging(debug=True)

directory = "../output/lenet1ep_seed4" #"../output/08-06-19_seed0"
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
    train={"n_epochs": 1} # 30
)(x)

# new 
y = Recipe(
    train={"n_epochs": 29}
)(x)

for _ in range(20):
    # finetune
    pruned = Recipe(
        prune_schema="../schemas/pruning_schema_lenet_unstructuredl1.py",
    )(x)
    x = Recipe(
        reinit_schema="../schemas/reinit_schema_lt_lenet.py",
        train={"n_epochs": 1}, # 30
    )(pruned) # LT

    # new
    y = Recipe(
        train={"n_epochs": 29},
    )(x)

    # x = Recipe(
    #     train={"n_epochs": 20},
    # )(pruned) # finetuning

experiment.run()