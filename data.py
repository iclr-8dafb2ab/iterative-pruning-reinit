import logging
import os
import torch
from torchvision import datasets, transforms

logger = logging.getLogger("data")

# TODO: get data transforms from here:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#load-data


def get_dataloaders(
    dataset,
    root=None,
    seed=0,
    device=torch.device("cpu"),
    train_batch_size=1,
    test_batch_size=1,
    num_workers=0,
    **kwargs,
):
    """
    Args:
        dataset: string, dataset identifier (only 'mnist' supported)
        root: string (optional), folder where the data will be downloaded.
        device: (optional) a torch.device() where the datasets will be located
                (default=torch.device('cpu'))
        seed: int (optional) manual seed used for torch and cudnn
        train_batch_size: int (optional), batch size for training dataloader
                          (default: 1)
        test_batch_size: int (optional), batch size for test dataloader
                          (default: 1)
        num_workers: int (optional), how many subprocesses to use for data 
                     loading. 0 means that the data will be loaded in the 
                     main process. (default: 0)
        **kwargs: allowed arguments include
            augment
            num_channels
            resize_to

    Returns:
        train_loader: pytorch DataLoader, training set
        test_loader: pytorch DataLoader, test set

    Raises:
        ValueError if dataset.lower is not one of {'mnist', 'cifar-10', 
            'cifar-100'}

    Notes:
    Make sure to read the documentation here (https://pytorch.org/docs/stable/data.html) 
    and here (https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed) 
    when using `num_workers` != 0, and pay particular attention to the role of random 
    seeds for reproducibility and workers syncing. 

    The training dataset will be shuffled (deterministically, depending on
    seed), while the test set will not be shuffled.
    """
    # set seeds and make sure results are deterministic (at least with
    # `num_workers` set to 0)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # determine transformations to apply depending on dataset nature
    if dataset.lower() == "mnist":
        # basic image standardization
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        if ("resize_to" in kwargs) and (kwargs["resize_to"] is not None):
            # TODO: check that resize_to makes sense
            resize_transform = [transforms.Resize(kwargs["resize_to"])]
            train_transform = transforms.Compose(
                resize_transform + train_transform.transforms
            )
            test_transform = transforms.Compose(
                resize_transform + test_transform.transforms
            )

        # data augmentation strategies
        if ("augment" in kwargs) and (kwargs["augment"] == True):
            augment_transforms = [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            train_transform = transforms.Compose(
                augment_transforms + train_transform.transforms
            )

        # if ("broadcast_channels" in kwargs) and (
        #     kwargs["broadcast_channels"] == True
        # ):
        if ("num_channels" in kwargs) and (kwargs["num_channels"] != 1):
            broadcast_transform = [
                transforms.Grayscale(
                    num_output_channels=kwargs["num_channels"]
                )
            ]
            train_transform = transforms.Compose(
                broadcast_transform + train_transform.transforms
            )
            test_transform = transforms.Compose(
                broadcast_transform + test_transform.transforms
            )

        root = root or "/anonymized/path/"

        try:
            train_dataset = datasets.MNIST(
                root=root,
                train=True,
                download=True,
                transform=train_transform,
            )
            test_dataset = datasets.MNIST(
                root=root,
                train=False,
                download=True,
                transform=test_transform,
            )
        except OSError:
            train_dataset = datasets.MNIST(
                root=os.path.expanduser("~"),
                train=True,
                download=True,
                transform=train_transform,
            )
            test_dataset = datasets.MNIST(
                root=os.path.expanduser("~"),
                train=False,
                download=True,
                transform=test_transform,
            )

    elif dataset.lower() in ["cifar-10", "cifar-100"]:

        # Construct initial transforms
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        if ("resize_to" in kwargs) and (kwargs["resize_to"] is not None):
            print(kwargs["resize_to"])
            resize_transform = [transforms.Resize(kwargs["resize_to"])]
            train_transform = transforms.Compose(
                resize_transform + train_transform.transforms
            )
            test_transform = transforms.Compose(
                resize_transform + test_transform.transforms
            )

        if ("augment" in kwargs) and (kwargs["augment"] == True):
            augment_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            train_transform = transforms.Compose(
                augment_transforms + train_transform.transforms
            )

        # if ("num_channels" in kwargs) and (kwargs["num_channels"] != 3):
        if "num_channels" in kwargs:
            if kwargs["num_channels"] == 3:
                norm_transform = [
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    )
                ]
                train_transform = transforms.Compose(
                    train_transform.transforms + norm_transform
                )
                test_transform = transforms.Compose(
                    test_transform.transforms + norm_transform
                )
            elif kwargs["num_channels"] == 1:
                broadcast_transform = [
                    transforms.Grayscale(
                        num_output_channels=kwargs["num_channels"]
                    )
                ]
                norm_transform = [transforms.Normalize((0.4790,), (0.2389,))]

                train_transform = transforms.Compose(
                    broadcast_transform
                    + train_transform.transforms
                    + norm_transform
                )
                test_transform = transforms.Compose(
                    broadcast_transform
                    + train_transform.transforms
                    + norm_transform
                )
            else:
                raise ValueError(
                    "Unexpected number of channels: {}".format(
                        kwargs["num_channels"]
                    )
                )

        if dataset.lower() in ["cifar-10"]:
            root = root or "/anonymized/path/"
            logger.info(
                "Data folder for {} dataset: {}".format(dataset, root)
            )

            try:
                train_dataset = datasets.CIFAR10(
                    root=root,
                    train=True,
                    download=True,
                    transform=train_transform,
                )
                test_dataset = datasets.CIFAR10(
                    root=root,
                    train=False,
                    download=True,
                    transform=test_transform,
                )
            except OSError:
                train_dataset = datasets.CIFAR10(
                    root=os.path.expanduser("~"),
                    train=True,
                    download=True,
                    transform=train_transform,
                )
                test_dataset = datasets.CIFAR10(
                    root=os.path.expanduser("~"),
                    train=False,
                    download=True,
                    transform=test_transform,
                )

        elif dataset.lower() == "cifar-100":
            root = root or "/anonymized/path/"
            logger.info(
                "Data folder for {} dataset: {}".format(dataset, root)
            )

            try:
                train_dataset = datasets.CIFAR100(
                    root=root,
                    train=True,
                    download=True,
                    transform=train_transform,
                )
                test_dataset = datasets.CIFAR100(
                    root=root,
                    train=False,
                    download=True,
                    transform=test_transform,
                )

            except OSError:
                train_dataset = datasets.CIFAR100(
                    root=os.path.expanduser("~"),
                    train=True,
                    download=True,
                    transform=train_transform,
                )
                test_dataset = datasets.CIFAR100(
                    root=os.path.expanduser("~"),
                    train=False,
                    download=True,
                    transform=test_transform,
                )

    else:
        raise ValueError(
            '`dataset` must be one of {"mnist, cifar-10, cifar-100"}'
        )

    # set up dataloaders from the datasets above, specifying the device,
    # sampling strategy, number of processes, etc.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    return train_loader, test_loader
