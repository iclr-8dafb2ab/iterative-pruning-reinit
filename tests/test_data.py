import pytest
import torch
import torchvision

import conftest
from data import get_dataloaders

class TestDataLoader():
    def test_dataset_type(self, dataset_name, device):
        """Test that the specified dataset is the one actually being loaded.
        """
        train_loader, test_loader = get_dataloaders(
            dataset_name,
            device=device,
            train_batch_size=3,
            test_batch_size=4
        )
        if dataset_name.lower() == 'mnist':
            assert isinstance(
                train_loader.dataset,
                torchvision.datasets.mnist.MNIST
            )
            assert isinstance(
                test_loader.dataset,
                torchvision.datasets.mnist.MNIST
            )
        elif dataset_name.lower() == 'cifar-10':
            assert isinstance(
                train_loader.dataset,
                torchvision.datasets.cifar.CIFAR10
            )
            assert isinstance(
                test_loader.dataset,
                torchvision.datasets.cifar.CIFAR10
            )
        elif dataset_name.lower() == 'cifar-100':
            assert isinstance(
                train_loader.dataset,
                torchvision.datasets.cifar.CIFAR100
            )
            assert isinstance(
                test_loader.dataset,
                torchvision.datasets.cifar.CIFAR100
            )
        else:
            raise ValueError("My test is flaky. Got unsupported dataset "
                "{} from conftest.".format(dataset_name))

    def test_wrong_dataset_name(self):
        """Calling `get_dataloaders` with a non-supported dataset_name will 
        raise a ValueError
        """
        with pytest.raises(ValueError):
            train_loader, test_loader = get_dataloaders(
                dataset='blah'
            )

    def test_dataset_successful_iter(self, dataset_name, device, num_channels):
        """Any error in the data preparation may cause runtime errors that 
        would not be visible otherwise
        """
        train_loader, test_loader = get_dataloaders(
            dataset_name,
            device=device,
            train_batch_size=3,
            test_batch_size=4,
            augment=True,
            resize_to=(48, 48),
            num_channels=num_channels,
        )
        x, y = iter(train_loader).next()
        assert True

    def test_cifar_wrong_numchannels(self, device):
        """num_channel=2 is invalid and should be caught
        """
        with pytest.raises(ValueError):
            train_loader, test_loader = get_dataloaders(
                dataset='cifar-10',
                device=device,
                train_batch_size=3,
                test_batch_size=4,
                augment=True,
                resize_to=(48, 48),
                num_channels=2,
            )

    def test_dataset_resize(self, dataset_name, device):
        """Test that passing resize_to actually resizes images to the correct 
        dimensions.
        """
        train_loader, test_loader = get_dataloaders(
            dataset_name,
            device=device,
            train_batch_size=3,
            test_batch_size=4,
            augment=True,
            resize_to=(48, 48),
            num_channels=1,
        )
        x, y = iter(train_loader).next()
        assert tuple(x.shape[-2:]) == (48, 48)

    def test_dataloader_gpu(self, dataset_name):
        """Specifying 'cuda' as the device will pin the memory.
        """
        train_loader, test_loader = get_dataloaders(
            dataset_name,
            device=torch.device('cuda'),
            train_batch_size=3,
            test_batch_size=4
        )
        assert train_loader.pin_memory
        assert test_loader.pin_memory

    def test_dataloader_batchsizes(self, dataset_name, device):
        """Test that the dataloaders yield batches of the requested sizes.
        """
        train_loader, test_loader = get_dataloaders(
            dataset_name,
            device=device,
            train_batch_size=3,
            test_batch_size=4
        )
        assert train_loader.batch_size == 3
        X_train_batch, y_train_batch = next(iter(train_loader))
        assert X_train_batch.shape[0] == 3
        assert X_train_batch.shape[0] == 3

        assert test_loader.batch_size == 4
        X_test_batch, y_test_batch = next(iter(test_loader))
        assert X_test_batch.shape[0] == 4
        assert X_test_batch.shape[0] == 4

    def test_dataloader_samplers(self, dataset_name, device):
        """The training set should be shuffled, while the test set should be 
        yielded in sequential, unshuffled order.
        """
        train_loader, test_loader = get_dataloaders(
            dataset_name,
            device=device,
            train_batch_size=3,
            test_batch_size=4
        )

        assert isinstance(
            train_loader.batch_sampler.sampler,
            torch.utils.data.sampler.RandomSampler
        )
        assert isinstance(
            test_loader.batch_sampler.sampler,
            torch.utils.data.sampler.SequentialSampler
        )

    def test_dataloader_determinism(self, dataset_name, device):
        """The randomness with which the training set is shuffled should be
        deterministic. So the order of the training batches is random but 
        constant.
        """
        train_loader1, _ = get_dataloaders(
            dataset_name,
            seed=0,
            device=device,
            train_batch_size=3,
            test_batch_size=4
        )
        X1, y1 = next(iter(train_loader1))

        train_loader2, _ = get_dataloaders(
            dataset_name,
            seed=0,
            device=device,
            train_batch_size=3,
            test_batch_size=4
        )
        X2, y2 = next(iter(train_loader2))

        assert torch.equal(X1, X2)
        assert torch.equal(y1, y2)

    def test_dataloader_nonempty(self, dataset_name, device):
        """Make sure that the dataloader contains a dataset of nonzero size
        """
        train_loader, test_loader = get_dataloaders(
            dataset_name,
            device=device,
            train_batch_size=3,
            test_batch_size=4
        )

        assert len(train_loader.dataset) > 0
        assert len(test_loader.dataset) > 0
