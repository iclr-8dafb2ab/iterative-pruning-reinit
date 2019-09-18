import pydoc
from torch import nn
import torch.nn.functional as F
import torchvision
import pruning as prune

# To be used to resize data to each model's expected input data size
MODEL_TO_INPUT_SIZE = {
    "torchvision.models.AlexNet": (224, 224),
    "models.LeNet": (28, 28),
}

MODEL_TO_NUM_CHANNELS = {"torchvision.models.AlexNet": 3, "models.LeNet": 1}


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model(model, *args, **kwargs):
    """
    model (string): model class in string format
    Raises:
        TypeError: if specified model class is not found
    Notes:
        from lottery_ticket code
    """
    # TODO: FIXME!
    # HACK: remove "models." from name to indicate models to find here
    if model.startswith("models."):
        model.replace(
            "models.", ""
        )  # HACK! Hoping it's the only "models." in name

    class_type = pydoc.locate(model)  # None if not found
    if not class_type:
        raise TypeError("Class {} not found".format(model))
    # instantiate model
    return class_type(*args, **kwargs)


def ensure_compatible_num_classes(model, num_classes):
    """
    Changes the output layer of the model in place, to match the number of 
    classes in the dataset, if needed. In that case, it wiill throw away 
    stored weights for last layer and reinitialize new layer.

    Note:
        It does nothing if the output layer already has the correct number
        of classes. CAUTION! Difference in behavior is huge.
        This is a bit of an ad-hoc procedure, as explained here:
        https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

    Args:
        model (pytorch nn.Module): model that needs to be modified
        num_classes (int): new number of classes

    Returns:
        None, modifies `model` in place, if needed.
    """
    # logger.info('Changing output layer to num_classes = {}'.format(num_classes))

    if type(model) == torchvision.models.AlexNet:
        if model.classifier[6].out_features == num_classes:
            return
        model.classifier[6] = nn.Linear(
            in_features=model.classifier[6].in_features,
            out_features=num_classes,
        ).to(device=model.classifier[6].weight.device)

    elif type(model) == LeNet:
        if model.fc3.out_features == num_classes:
            return
        model.fc3 = nn.Linear(
            in_features=model.fc3.in_features, out_features=num_classes
        ).to(device=model.fc3.weight.device)


def is_pruned(model):
    """Check whether `model` is pruned by looking for forward_pre_hooks in its
    modules that inherit from the BasePruningMethod.

    Args:
        model (torch.nn.Module): object that is either pruned or unpruned

    Returns:
        binary answer to whether `model` is pruned.

    TODO: test
    """
    for _, module in model.named_module():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, prune.BasePruningMethod):
                return True
    return False
