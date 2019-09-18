import torch
import logging
from initializations import get_matching_parameter

logger = logging.getLogger("experiment_pipeline")


def train_model(
    model,
    loader,
    n_epochs,
    loss_func,
    optimizer,
    callbacks,
    device=torch.device("cuda"),
):
    """Main model training loop, in which `model` is trained for `n_epochs` on
    data from the `loader`, using `loss_func` as a loss function to be
    minimized by `optimizer`. At each iteration, the list of actions
    determined by the `callbacks` is performed.

    Args:
        model (torch.nn.Module): model to train
        loader (torch.utils.data.DataLoader): training data
        n_epochs (int): number of training epochs
        loss_func (torch.nn.modules.loss): objective function
        optimizer (torch.optim): optimizer
        callbacks (list): actions to perform at every training iteration (e.g.
            logging, checkpointing, evaluating, plotting, ...)
        device: TODO
    """
    # Training loop
    for epoch_n in range(n_epochs):
        for batch_n, (X, y) in enumerate(loader):
            X = X.to(device=device)
            y = y.to(device=device)
            model = model.to(device=device)
            # train
            optimizer.zero_grad()
            yhat = model(X)
            loss = loss_func(yhat, y)
            loss.backward()
            optimizer.step()

            for c in callbacks:
                c(epoch_n, batch_n, model, optimizer)
            # # evaluate
            # # checkpoint


def prune_model(model, pruning_schema, seed):
    """Use the instructions in the pruning schema to apply the correct pruning
    function to the correct tensor within the correct module.
    This uses `named_modules` to get the module object.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        model.to(device=torch.device("cuda"))

    for (module_name, tensor_name), pruning_fn in pruning_schema.items():
        # use the module name to extract the module from the model
        if module_name not in dict(model.named_modules()):
            raise KeyError(
                "Module {} not found. Available modules: {}".format(
                    module_name, dict(model.named_modules()).keys()
                )
            )
        module = dict(model.named_modules())[module_name]
        # now that we have both module and tensor_name, we prune them
        pruning_fn(module, tensor_name)


def init_model(model, init_schema, seed):
    """Use the instructions in the init schema to apply the correct 
    initialization function to the correct tensor within the model.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    for (module_name, tensor_name), init_fn in init_schema.items():
        # use the module name to extract the module from the model
        parameter_name = ".".join((module_name, tensor_name))
        tensor = get_matching_parameter(parameter_name, model)
        # now that we have both module and tensor_name, we prune them
        init_fn(tensor)
