import pytest
import torch

import conftest
from optimizers import get_optimizer
from models import get_model


class TestOptimizer:
    def test_optimizer_type(self, optimizer_name, model_name):
        """Test that the specified optimizer is the one actually being loaded.
        """
        model = get_model(model_name)
        optimizer = get_optimizer(
            optimizer_name, model.parameters(), lr=0.001
        )
        if optimizer_name.lower() == "sgd":
            assert isinstance(optimizer, torch.optim.SGD)
        elif optimizer_name.lower() == "adam":
            assert isinstance(optimizer, torch.optim.Adam)
        else:
            raise ValueError(
                "My test is flaky. Got unsupported optimizer "
                "{} from conftest.".format(optimizer_name)
            )

    def test_wrong_optimizer_name(self, model_name):
        """Calling `get_optimizer` with a non-supported optimizer_name will 
        raise a ValueError
        """
        model = get_model(model_name)
        with pytest.raises(ValueError):
            optimizer = get_optimizer("blah", model.parameters(), lr=0.001)
