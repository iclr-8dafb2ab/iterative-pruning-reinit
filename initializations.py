import torch
import torch.nn.init as init
import pickle
import pruning as prune


def non_constant_(tensor, val):
    init.constant_(tensor, val)
    tensor.data[0] = val - 3


def init_copy_(tensor, other):
    """Reinitializes `tensor` in place with the values contained in `other`.

    Args:
        tensor (torch.Tensor): tensor that gets reinit in place
        other (torch.Tensor): tensor whose values get copied into `tensor`
    """
    if tensor.size() != other.size():
        raise ValueError(
            "The two tensors need to have matching size. Found "
            "tensor of size {} and other of size {}.".format(
                tensor.size(), other.size()
            )
        )
    with torch.no_grad():
        tensor.copy_(other)


def zhou_initialization(curr_param, param_name, experiment_state):
    # NOTE THIS CALCULATES THE SAMPLE STANDARD DEVIATION
    # See section 3 of Zhou et al 2019
    from dag import ExperimentState

    if (
        not hasattr(experiment_state, "root")
        or not hasattr(experiment_state.root, "path")
        or experiment_state.root.path is None
    ):
        raise AttributeError("root path not set for current experiment state")

    initialized = ExperimentState.load(experiment_state.root.path)

    init_param = get_matching_parameter(param_name, initialized.model)
    init_copy_(curr_param, init_param.std().item() * torch.sign(init_param))


def sign_initialization(curr_param, param_name, experiment_state):
    from dag import ExperimentState

    if (
        not hasattr(experiment_state, "root")
        or not hasattr(experiment_state.root, "path")
        or experiment_state.root.path is None
    ):
        raise AttributeError("root path not set for current experiment state")

    initialized = ExperimentState.load(experiment_state.root.path)

    init_param = get_matching_parameter(param_name, initialized.model)
    init_copy_(curr_param, torch.sign(init_param))


def rewind_initialization(curr_param, param_name, experiment_state):
    """Reinitializes the value of curr_param in the current model using
    the values of that parameter found in the model stored in the
    `experiment_state.root.path`.

    Args:
        curr_param:
        experiment_state (ExperimentState): current state whose model will be
            reinitialized.
    """
    from dag import ExperimentState

    # check first of all that root.path is even set for experiment_state
    if (
        not hasattr(experiment_state, "root")
        or not hasattr(experiment_state.root, "path")
        or experiment_state.root.path is None
    ):
        raise AttributeError("root path not set for current experiment state")

    # if it's set, then load it back on
    initialized = ExperimentState.load(experiment_state.root.path)

    init_param = get_matching_parameter(param_name, initialized.model)
    init_copy_(curr_param, init_param)


def get_matching_parameter(param_name, curr_model):
    """Abstracts away the parameter matching portion of any reinitialization,
    by handling the three possible cases:
        1) all parameter names match between the current and initial model
        2) the unpruned version of the parameter exists in the initial model
            but its pruned version exists in the current model
        3) ^ the opposite of 2)
    This is needed because pruned models in which the pruning
    reparametrization is not removed have parameters named <name>+"_orig".

    Raises:
        KeyError: if any `param_name` found in the initial model has no
            equivalent in the `curr_model`.

    Args:
        param_name (string): name of the parameter in the reference model
        curr_model (torch.nn.Module): model in which we wish to find the
            parameter.

    Returns:
        curr_param (torch.nn.parameter.Parameter): parameter in `curr_model`
            that ~matches `param_name`.
    """
    curr_parameters = dict(curr_model.named_parameters())

    # handle all of the identical parameters normally
    if param_name in curr_parameters:
        curr_param = curr_parameters[param_name]
    # handle the params that are unpruned in init but pruned in curr
    elif (param_name + "_orig") in curr_parameters:
        curr_param = curr_parameters[param_name + "_orig"]
    # handle the params that are unpruned in init but pruned in curr
    elif ("_orig" in param_name) and (
        param_name.replace("_orig", "") in curr_parameters
    ):
        curr_param = curr_parameters[param_name.replace("_orig", "")]
    else:
        raise KeyError(
            "Found param {} in the initial model ".format(param_name)
            + "with no equivalent in the current model. These are the "
            "params in the current model: {}".format(curr_parameters.keys())
        )
    return curr_param
