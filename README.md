# Science of Neural Network Pruning
An experimental pipeline to run careful experimentation around the role, biases, and effects of neural network pruning on learning dynamics, performance metrics, internal concept representation, and more.

## Main Idea
Inspired by the rapid pace of research results around pruning techniques for deep, over-parameterized neural networks, we developed a setup for proper, careful experimentation to get at a fundamental understanding of the effects of pruning on various aspects of neural network training. This work allows for flexible and quick experimental changes to be implemented and tested to get high-quality, trustworthy, scientific results that control for other confounding factors and nuisance parameters.

## Getting Started

Install all requirements (preferably in a virtualenv!) by running `pip install -r requirements.txt`.

## Code Structure
The core primitives for creating immutable experiment states are contained in [`dag.py`](dag.py), which is the core file that connects all other implemented functionalities into one single pipeline. 

The example file [`sandbox.py`](sandbox.py) provides an example of how all meaningful parts can be organized into a logical workflow for pruning experimentation. To design different experiments which make use of alternative training/pruning loops, modify this file to implement your logic.

**The key entities to know are: `ExperimentState` and `Recipe`.** An `ExperimentState` represents the notion of frozen point in time in the lifecycle of a neural net -- this could be post-initialization, after an epoch of training, or after pruning with *L1*-structured pruning. To transition between states, we use a `Recipe`, which represents *how* to transition, whether via pruning, reinitialization, fine-tuning, or any combination thereof.

The functional view of recipes being applied to states leads to the notion of a tree-structure that represents how an experiment may evolve. For example, we may perform the following, where we do two rounds of lottery ticket reinitialization:


```
/-------------------\
| Initialized LeNet |
\-------------------/
     |    |    |
     |    |     \-> TRAIN -> [state] -> PRUNE -> [state]
     |    |
     |    |                                         |
     |    |                                         v
     |    |
     |    \-------------------------------------> REINIT -> TRAIN -> [state] -> PRUNE -> [state]
     |     
     |                                                                                      |
     |                                                                                      v  
     |     
     \----------------------------------------------------------------------------------> REINIT
``` 

Experiment status tracking, handling, and saving is taken care of by the `ExperimentState` class in [`dag.py`](dag.py). The logic used in the pipeline is that the initial experiment state will serve as the root of the tree of experiments that will spring from the same initialization. Indeed, one can chose different training and pruning strategies, all of which can share the same initial state. Each new `ExperimentState` will then be linked to its parent via the `parent` attribute.

All fundamental building blocks of a pruning experiment are independently implemented as modular functions in the following files:
* [`data.py`](data.py) implements data loading, transformation, sampling, plotting, investigation, ...
* [`dag.py`](dag.py) implements experiment management primitives.
* [`models.py`](models.py) fetches models, defines architecture, ...
* [`plotting.py`](plotting.py) provides useful functions to plot network weights and biases in conv2d and linear layers as well as visualize the experiment graph.

Finally, [`utils.py`](utils.py) handles logging.

## Running Example

Simply run `python3 sandbox.py`.


## Running the experiments

The `recipes/` directory contains all pertainent experiments, with each runnable as a standalone script. From the `recipes/` directory, run `python3 recipe_{whichever}.py`, which will run the experiment, and deposit the trained experiment graph in `../output`. For example, `recipe_lt_structuredL1.py` runs *L1*-structured pruning for lottery tickets.

## Running the Tests

You can run all the tests using the command: `pytest -v tests/*`

## Determinism and Reproducibility

Stochasticity can enter these experiments in many different ways. It is important to make use of explicit seeds and unit testing to ensure full reproducibility. 

In [`data.py`](data.py), for data loading and batch yielding, the test set is yielded in a deterministic, unshuffled manner using a `SequentialSampler`, while the train set is yielded in a deterministic, yet randomly shuffled manner using a `RandomSampler`. The function `get_dataloaders` that returns the train and test dataloaders takes in and sets a random seed, which is key to ensure determinism. (That, however, doesn't handle advancing the loader to a specific iteration in a cleaner way than just hitting "next" until the right iteration is reached).

## License
