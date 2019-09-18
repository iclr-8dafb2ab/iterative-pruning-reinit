import torch
import dask
import logging
import uuid
from functools import partial
import pathlib
import dill
import hashlib

from data import get_dataloaders
import initializations
from models import (
    MODEL_TO_INPUT_SIZE,
    MODEL_TO_NUM_CHANNELS,
    LeNet,
    get_model,
    ensure_compatible_num_classes,
)
from optimizers import get_optimizer
import pruning as prune
from experiment_pipeline import train_model, prune_model, init_model

logger = logging.getLogger("DAG")


def stdout_logging_callback(epoch_n, batch_n, model, optimizer):
    """
    N.B., THIS IS MEANT TO BE TEMPORARY.
    """
    if not batch_n % 100:
        logger.debug(f"Epoch {epoch_n}, batch {batch_n}")


class Experiment:
    """
    An Experiment object retains state for all the ExperimentStates and
    subsequent transitions that are governed by Recipe objects.

    Attributes:
        directory (pathlib.Path): The directory all the stages of the
        experiment will be stored in.
        leaves (Dict[uuid.UUID, dask.Delayed]): (internal) Stores the final
            states for leaf nodes in the Experiment graph. Dask will traverse
            until the leaf nodes are computed, in the case where we lazily
            define the graph.
        nodes (Dict[uuid.UUID, dask.Delayed]): (internal) Stores the final
            states for all nodes in the Experiment graph. 
        root (ExperimentState): The (never changing) root of the
            Experiment graph.
    """

    def __init__(self, directory):
        """Create a new Experiment object.

        An Experiment provides a sort-of anchor for a DAG of experimental steps.
        To create an Experiment, you must provide a directory, which will be the
        location for the Experiment to deposit a file for each state in the
        experiment graph.

        Args:
            directory (str | pathlib.Path): Path to a directory (will be
                created) which will be the root for all experment state
                transitions to be stored.
        """
        self.nodes = {}
        self.leaves = {}
        self.root = None
        self.directory = pathlib.Path(directory).absolute()
        self.directory.mkdir(exist_ok=True)

    def spawn_new_tree(
        self,
        dataset_name,
        model_name,
        init_schema,
        optimizer_name,
        seed,
        device=torch.device("cuda"),
    ):
        """
        Create the initial experiment state to anchor off of the current
        Experiment object.

        Args:
            dataset_name (str): Name of dataset to load in.
            model_name (str): Name of model to load in.
            init_schema: TODO
            optimizer_name (str): Name of optimizer to use in training.
            seed: pytorch seed to ensure determinism in training dataloader
            device: TODO

        Returns:
            ExperimentStatePromise: The initial state represented as a promise.

        Raises:
            RuntimeError: If the Experiment object already has a root
                ExperimentState.

        TODO: will take in dataset_hparams and optimizer_hparams instead of 
            just their names.
        """
        if self.root:
            raise RuntimeError(
                "This Experiment object already has a root ExperimentState!"
            )

        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warn("Falling back to CPU: no GPU detected")
            device = torch.device("cpu")

        # set seeds for determinism
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        # Here, we begin defining what should be contained in an
        # ExperimentState. The main thing to note here is that an
        # ExperimentState is initialized with a parent_path (a file pointer to
        # it's progenitor state) and the experiment object itself. This is
        # critical so every ExperimentState in an Experiment knows where it
        # comes from and can find its root.
        initial_state = ExperimentState(
            parent_path=None, experiment_object=self, device=device
        )

        # Just store these attributes so we have them around. We also use them
        # to "hash" the state so we can always reload an existing state in a
        # graph.
        initial_state.dataset_name = dataset_name
        initial_state.model_name = model_name
        initial_state.optimizer_name = optimizer_name

        # A Recipe is the way in which an ExperimentState moves to a new
        # ExperimentState, i.e., the state transitions. A Recipe can train,
        # prune, and reinitialize a network (it can do any subset of these
        # three, in fact). Here, we track the arguments that are used for each
        # stage in the recipe, and we store these in the **resultant**
        # ExperimentState so we can hash them into the representation of the
        # state.
        if init_schema:
            init_schema = open(init_schema).read()
        initial_state.train_recipe_args = {"init_schema": init_schema}
        initial_state.prune_recipe_args = ""
        initial_state.reinit_recipe_args = ""

        # TODO: Do we want to have these passed in somehow?
        initial_state.dataloader_hparams = {
            "dataset": dataset_name,
            "root": "./data",
            "seed": seed,
            "device": torch.device("cuda"),
            "train_batch_size": 32,
            "test_batch_size": 32,
            "num_workers": 0,
            "augment": True,
            # "broadcast_channels": False,
            "resize_to": MODEL_TO_INPUT_SIZE[model_name],
            "num_channels": MODEL_TO_NUM_CHANNELS[model_name],
        }

        # Add dataloaders to the state.
        (
            initial_state.train_dataloader,
            initial_state.test_dataloader,
        ) = get_dataloaders(**initial_state.dataloader_hparams)

        # Instantiate model
        model = get_model(model_name)

        # Modify last layer of model to fit data
        ensure_compatible_num_classes(
            model=model,
            num_classes=len(initial_state.train_dataloader.dataset.classes),
        )

        if init_schema:
            init_model(model, eval(init_schema), seed=seed)

        # Finally, add the model to the state (not a copy, of course)
        initial_state.model = model

        # TODO: Same here as above with dataloader hparams, should these be
        # arguments to the function that initializes the Experiment root?
        initial_state.optimizer_hparams = {"lr": 0.01}
        initial_state.optimizer = get_optimizer(
            optimizer_name,
            initial_state.model.parameters(),
            **initial_state.optimizer_hparams,
        )

        initial_state_path = self.directory / (
            initial_state.filename() + "-root"
        )

        # We construct the state, and check and see if we already have an
        # experiment state loaded for the root. If we do, we load it.
        if initial_state_path.exists():
            logger.info("Root node of experiment exists! Reloading state.")
            initial_state = ExperimentState.load(initial_state_path)
            initial_state.from_cache = True
            initial_state.experiment_object = self
        else:
            logger.info("New root node of experiment, saving.")
            initial_state.save(initial_state_path)

        self.root = initial_state
        self.root.seed = seed

        return ExperimentStatePromise(
            promise=initial_state,
            id=uuid.uuid4(),
            previous_id=None,
            experiment_object=self,
        )

    def run(self, scheduler="single-threaded"):
        """Run the Experiment defined by the graph.

        Args:
            scheduler (str): How dask should schedule the nodes in the graph
                (see the dask documentation for more information).
        """
        _ = dask.compute(self.leaves, scheduler=scheduler)
        # when dask goes thru the tree, it knows the full sequence of ops
        # needed to compute each leaf, so this gives dask full authority in
        # determining the best dispatch path.


class ExperimentStatePromise:
    """An ExperimentStatePromise is a construct to allow for a lazy graph.

    Specifically, an ExperimentStatePromise encapsulates a lazy-evaluated
    function that represents the transition from state to state.

    Attributes:
        experiment_object (Experiment): The anchor Experiment for this promise.
        id (uuid.UUID): Unique ID of this experiment state promise.
        previous_id (uuid.UUID): ID of the state which directly feeds into the
            next state defined by the realization of the promise.
        promise (object): The object to be lazily acted upon.
    """

    def __init__(self, promise, id, previous_id, experiment_object):
        """Creates a new promose from an existing object.

        Args:
            experiment_object (Experiment): The anchor Experiment for
                this promise.
            id (uuid.UUID): Unique ID of this experiment state promise.
            previous_id (uuid.UUID): ID of the state which directly feeds
                into the next state defined by the realization of the promise.
            promise (object): The object to be lazily acted upon.
        """
        self.promise = promise
        self.id = id
        self.previous_id = previous_id
        self.experiment_object = experiment_object

    def promise_from_callable(self, fn):
        """Defines the function which is to act on the promise which evolves
        the object from state A to state B.

        Args:
            fn (Callable): Function that takes as input an object of whatever
                type the promise is and returns a python object.

        Returns:
            ExperimentStatePromise
        """
        return ExperimentStatePromise(
            promise=dask.delayed(fn)(self.promise),
            id=uuid.uuid4(),
            previous_id=self.id,
            experiment_object=self.experiment_object,
        )

    def get(self, scheduler="single-threaded"):
        # this is for all the lazily evaluated states
        if hasattr(self.promise, "compute"):
            return self.promise.compute(scheduler=scheduler)
        # this is for the initial state that materializes immediately
        else:
            return self.promise


class ExperimentState:
    """An ExperimentState represents a point in time with the evolution of an
    experiment. An ExperimentState can have any number of objects attached to
    it as attributes. Importantly, we must have a consistent way to save a
    state such that it can be reloaded if an identical experiment is to be
    run (cacheing).

    Attributes:
        experiment_object (Experiment): The anchor experiment for the state.
            parent_path (pathlib.Path): Path to the saved version of the
            parent state.
    """

    def __init__(
        self, parent_path, experiment_object, device=torch.device("cuda")
    ):
        # super(ExperimentState, self).__init__()
        self.parent_path = parent_path
        self.experiment_object = experiment_object
        self.path = None
        self.from_cache = False

        # If the experiment state has a parent, the following attributes will
        # be inhereted these are shared across all states that stem from the
        # same root
        if self.parent_path is not None:
            parent = self.__class__.load(self.parent_path)
            for property_name in [
                "train_dataloader",
                "test_dataloader",
                "model",
                "optimizer",
                "dataset_name",
                "model_name",
                "optimizer_name",
                "dataloader_hparams",
                "optimizer_hparams",
            ]:
                value = getattr(parent, property_name, None)
                setattr(self, property_name, value)
            if device.type == "cuda":
                self.move_model(device=device)

        self.device = device

    @property
    def root(self):
        return self.experiment_object.root

    def filename(self):
        obj = {
            "parent_path": self.parent_path,
            "path": self.path,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "optimizer_name": self.optimizer_name,
            "train_recipe_args": str(
                sorted(list(self.train_recipe_args.items()))
            ),
            "prune_recipe_args": self.prune_recipe_args,
            "reinit_recipe_args": self.reinit_recipe_args,
            "dataloader_hparams": str(
                sorted(list(self.dataloader_hparams.items()))
            ),
            "optimizer_hparams": str(
                sorted(list(self.optimizer_hparams.items()))
            ),
        }
        representation = str(sorted(list(obj.items())))
        # TODO: Make the representation better
        h = hashlib.md5()
        h.update(representation.encode())
        return h.hexdigest()

    def __getstate__(self):
        """
        The getstate hook is invoked by dill and pickle - we don't want to
        serialize the experiment object!
        """
        o = dict(self.__dict__)
        del o["experiment_object"]
        return o

    def __setstate__(self, s):
        """
        The setstate hook is invoked by dill and pickle - we don't want to
        serialize the experiment object!
        """
        self.__dict__ = s
        self.experiment_object = None

    def move_model(self, device):
        # Move model back to cpu
        self.model = self.model.to(device)
        # Manually move weights for forward (from pruning) to cpu
        layers = [
            layer
            for layer in dict(self.model.named_modules()).values()
            if not list(layer.named_children())
        ]
        for layer in layers:
            for (attr_name, attribute) in layer.__dict__.items():
                if type(attribute) == torch.Tensor:
                    setattr(
                        layer,
                        attr_name,
                        getattr(layer, attr_name).to(device=device),
                    )
        #                     attribute.to(device=torch.device("cpu"))
        #                     setattr(layer, attr_name, attribute.cpu())
        if device == torch.device("cpu"):
            torch.cuda.empty_cache()

    def save(self, path):
        # always save out to cpu
        self.move_model(device=torch.device("cpu"))
        self.path = pathlib.Path(path)
        dill.dump(self, open(self.path, "wb"))

    @classmethod
    def load(cls, path, device=torch.device("cpu")):
        state = dill.load(open(path, "rb"))
        state.move_model(device)
        state.path = pathlib.Path(state.path)
        return state


class Recipe:
    """A Recipe represents a sequence of actions that modify an 
    ExperimentState. The supported actions currently include: training a 
    model, pruning a model, and reinitializing a model. The instructions on
    how to perform these actions are contained in the train, prune_schema, and
    reinit_schema attributes set at init (see sandbox.py).
    """

    def __init__(
        self,
        *,  # Ensure the following args can't be positional args
        train=None,
        prune_schema=None,
        reinit_schema=None,
        name=None,
        seed=None,
        **extra_kwargs,
    ):
        self.name = name
        self.train = train
        self.prune_schema = prune_schema
        self.reinit_schema = reinit_schema
        self.extra_kwargs = extra_kwargs
        self.seed = seed

    def __call__(self, experiment_state):
        """This is what adds the ops defined by the recipe to the graph
        """
        if self.seed is None:
            self.seed = experiment_state.experiment_object.root.seed

        new_state = experiment_state.promise_from_callable(
            partial(
                self.run,
                train=self.train,
                prune_schema=self.prune_schema,
                reinit_schema=self.reinit_schema,
            )
        )

        # If the previous state (`experiment_state`) was a leaf, but we now
        # created a new state stemming from it, then the previous state can
        # be removed from the set of leaves.
        if experiment_state.id in experiment_state.experiment_object.leaves:
            del experiment_state.experiment_object.leaves[experiment_state.id]

        # The new state can now be added to the leaves
        new_state.experiment_object.leaves[new_state.id] = new_state.promise
        # Add it to the nodes as well
        new_state.experiment_object.nodes[new_state.id] = new_state.promise
        return new_state

    def run(
        self,
        experiment_state,
        train=None,
        prune_schema=None,
        reinit_schema=None,
    ):

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True

        if self.name:
            logger.info("Running Recipe {}".format(self.name))

        new_state = ExperimentState(
            parent_path=experiment_state.path,
            experiment_object=experiment_state.experiment_object,
            device=experiment_state.device,
        )

        # We'll probably need to only select out the arguments that are
        # interesting for determining whether or not an experiment has been
        # performed before (for example, printing & verbosity information
        # probably shouldn't determine whether or not a previously run subset
        # of an experiment should be reloaded or not).
        # N.B., these arguments are critical for looking up to see whether or
        # not a subset of an experimental pipeline has been previously run.
        if prune_schema:
            prune_schema = open(prune_schema).read()
        if reinit_schema:
            reinit_schema = open(reinit_schema).read()
        new_state.train_recipe_args = dict(train or {})
        new_state.prune_recipe_args = prune_schema or ""
        new_state.reinit_recipe_args = reinit_schema or ""
        # TODO: normalize python code inside using ast

        # Define a path to the new state of the experiment. The name is added
        # to the end to help with human readability of the state files in the
        # experiment's directory.
        new_state_path = experiment_state.experiment_object.directory / (
            new_state.filename()
            + (f"-{self.name}" if self.name else "")
            + (f"-{self.seed}" if self.name else "")
        )

        # This is now the case where we've actually done the experiment, and we
        # can safely just load the state that we had before
        if new_state_path.exists():
            logger.info(
                "State already exists! Safely loading existing state."
            )
            new_state = ExperimentState.load(new_state_path)
            new_state.from_cache = True
            # Reset exp object because it gets deleted from state before
            # serialization
            new_state.experiment_object = experiment_state.experiment_object

        # Here, we haven't run this setup before, so we need to go ahead and
        # actually run the experiment.
        else:
            logger.info(
                "No matching experiment state exists, transitioning state now."
            )

            # In this order: prune, reinit, train
            if prune_schema is not None:
                prune_model(
                    new_state.model,
                    pruning_schema=eval(prune_schema),
                    seed=self.seed,
                )
                logger.info("Pruning complete")

            if reinit_schema is not None:
                # this allows us to reset to the original initialization
                global_ctx = globals()
                global_ctx.update({"current_state": new_state})

                init_model(
                    model=new_state.model,
                    init_schema=eval(reinit_schema, global_ctx, locals()),
                    seed=self.seed,
                )
                logger.info("Reinitialization complete")

            if train is not None:
                if not isinstance(train, dict):
                    raise TypeError(
                        "`train` must be a dictionary containing arguments "
                        "that are passed to the `train` function."
                    )
                logger.info("Training as part of this recipe.")
                try:
                    n_epochs = int(train["n_epochs"])
                except KeyError as err:
                    raise KeyError(
                        "Missing required argument `n_epochs` in arguments "
                        "to `train` function."
                    )

                # TODO: Decide how to handle arguments related to loss function
                # and to verbosity, callbacks, etc.
                train_model(
                    model=new_state.model,
                    loader=new_state.train_dataloader,
                    n_epochs=n_epochs,
                    loss_func=torch.nn.CrossEntropyLoss(),
                    optimizer=new_state.optimizer,
                    callbacks=[stdout_logging_callback],
                    device=new_state.device,
                )
                logger.info("Training complete")

            new_state.save(new_state_path)

        return new_state
