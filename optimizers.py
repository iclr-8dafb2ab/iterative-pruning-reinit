import torch


def get_optimizer(optimizer_name, parameters, **kwargs):
    # TODO: support 'custom' to support per-param options
    # see https://pytorch.org/docs/stable/optim.html#per-parameter-options
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(parameters, **kwargs)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(parameters, **kwargs)
    else:
        raise ValueError("`optimizer_name` must be one of {'adam', 'sgd'}")

    return optimizer
