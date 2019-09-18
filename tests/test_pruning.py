"""
Adapted from pytorch unit tests for pruning functinalities.
"""
from unittest import mock
import pickle
import pytest
import torch
from torch import nn

import pruning
from pruning import (
    identity,
    random_unstructured,
    l1_unstructured,
    remove,
    random_structured,
    ln_structured,
    global_unstructured,
    _validate_pruning_amount_init,
    _validate_pruning_amount,
    _compute_nparams_toprune,
    L1PruningMethod,
    RandomPruningMethod,
    LnStructuredPruningMethod,
    PruningContainer,
    CustomFromMaskPruningMethod,
    RandomStructuredPruningMethod
)

# TODO: add other modules to parametrized tests
class TestPruning:
    def test_validate_pruning_amount_init(self):
        """Test the first util function that validates the pruning
        amount requested by the user the moment the pruning method
        is initialized. This test checks that the expected errors are
        raised whenever the amount is invalid.
        The orginal function runs basic type checking + value range checks.
        It doesn't check the validity of the pruning amount with
        respect to the size of the tensor to prune. That's left to
        `_validate_pruning_amount`, tested below.
        """
        # neither float not int should raise TypeError
        with pytest.raises(TypeError):
            _validate_pruning_amount_init(amount="I'm a string")

        # float not in [0, 1] should raise ValueError
        with pytest.raises(ValueError):
            _validate_pruning_amount_init(amount=1.1)
        with pytest.raises(ValueError):
            _validate_pruning_amount_init(amount=20.0)

        # negative int should raise ValueError
        with pytest.raises(ValueError):
            _validate_pruning_amount_init(amount=-10)

        # all these should pass without errors because they're valid amounts
        _validate_pruning_amount_init(amount=0.34)
        _validate_pruning_amount_init(amount=1500)
        _validate_pruning_amount_init(amount=0)
        _validate_pruning_amount_init(amount=0.0)
        _validate_pruning_amount_init(amount=1)
        _validate_pruning_amount_init(amount=1.0)
        assert True

    def test_validate_pruning_amount(self):
        """Tests the second util function that validates the pruning
        amount requested by the user, this time with respect to the size
        of the tensor to prune. The rationale is that if the pruning amount,
        converted to absolute value of units to prune, is larger than
        the number of units in the tensor, then we expect the util function
        to raise a value error.
        """
        # if amount is int and amount > tensor_size, raise ValueError
        with pytest.raises(ValueError):
            _validate_pruning_amount(amount=20, tensor_size=19)

        # amount is a float so this should not raise an error
        _validate_pruning_amount(amount=0.3, tensor_size=0)

        # this is okay
        _validate_pruning_amount(amount=19, tensor_size=20)
        _validate_pruning_amount(amount=0, tensor_size=0)
        _validate_pruning_amount(amount=1, tensor_size=1)
        assert True

    def test_compute_nparams_toprune(self):
        """Test that requested pruning `amount` gets translated into the
        correct absolute number of units to prune.
        """
        assert _compute_nparams_toprune(amount=0, tensor_size=15) == 0
        assert _compute_nparams_toprune(amount=10, tensor_size=15) == 10
        # if 1 is int, means 1 unit
        assert _compute_nparams_toprune(amount=1, tensor_size=15) == 1
        # if 1. is float, means 100% of units
        assert _compute_nparams_toprune(amount=1.0, tensor_size=15) == 15
        assert _compute_nparams_toprune(amount=0.4, tensor_size=17) == 7

    @pytest.mark.parametrize("m", [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)])
    @pytest.mark.parametrize("name", ["weight", "bias"])
    def test_random_unstructured_sizes(self, m, name):
        """Test that the new parameters and buffers created by the pruning
        method have the same size as the input tensor to prune. These, in
        fact, correspond to the pruned version of the tensor itself, its
        mask, and its original copy, so the size must match.
        """
        original_tensor = getattr(m, name)

        random_unstructured(m, name=name, amount=0.1)
        # mask has the same size as tensor being pruned
        assert original_tensor.size() == getattr(m, name + "_mask").size()
        # 'orig' tensor has the same size as the original tensor
        assert original_tensor.size() == getattr(m, name + "_orig").size()
        # new tensor has the same size as the original tensor
        assert original_tensor.size() == getattr(m, name).size()

    @pytest.mark.parametrize("m", [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)])
    @pytest.mark.parametrize("name", ["weight", "bias"])
    def test_random_unstructured_orig(self, m, name):
        """Test that original tensor is correctly stored in 'orig'
        after pruning is applied. Important to make sure we don't
        lose info about the original unpruned parameter.
        """
        original_tensor = getattr(m, name)  # tensor prior to pruning
        random_unstructured(m, name=name, amount=0.1)
        assert torch.equal(original_tensor, getattr(m, name + "_orig"))

    @pytest.mark.parametrize("m", [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)])
    @pytest.mark.parametrize("name", ["weight", "bias"])
    def test_random_unstructured_new_weight(self, m, name):
        """Test that module.name now contains a pruned version of
        the original tensor obtained from multiplying it by the mask.
        """
        original_tensor = getattr(m, name)  # tensor prior to pruning
        random_unstructured(m, name=name, amount=0.1)
        # weight = weight_orig * weight_mask
        assert torch.equal(
            getattr(m, name),
            getattr(m, name + "_orig")
            * getattr(m, name + "_mask").to(dtype=original_tensor.dtype),
        )

    def test_identity(self):
        """Test that a mask of 1s does not change forward or backward.
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        assert torch.equal(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        assert torch.equal(old_grad_bias, torch.ones_like(m.bias))

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        identity(m, name="weight")

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        assert torch.equal(y_prepruning, y_postpruning)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        assert torch.equal(old_grad_weight, m.weight_orig.grad)
        assert torch.equal(old_grad_bias, m.bias.grad)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        assert torch.equal(y1, y2)

    def test_random_unstructured_0perc(self):
        """Test that a mask of 1s does not change forward or backward.
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)
        y_prepruning = m(input_)  # output prior to pruning

        # compute grad pre-pruning and check it's equal to all ones
        y_prepruning.sum().backward()
        old_grad_weight = m.weight.grad.clone()  # don't grab pointer!
        assert torch.equal(old_grad_weight, torch.ones_like(m.weight))
        old_grad_bias = m.bias.grad.clone()
        assert torch.equal(old_grad_bias, torch.ones_like(m.bias))

        # remove grads
        m.zero_grad()

        # force the mask to be made of all 1s
        with mock.patch(
            "pruning.RandomPruningMethod.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = torch.ones_like(m.weight)
            random_unstructured(m, name="weight", amount=0.9)  # amount won't count

        # with mask of 1s, output should be identical to no mask
        y_postpruning = m(input_)
        assert torch.equal(y_prepruning, y_postpruning)

        # with mask of 1s, grad should be identical to no mask
        y_postpruning.sum().backward()
        assert torch.equal(old_grad_weight, m.weight_orig.grad)
        assert torch.equal(old_grad_bias, m.bias.grad)

        # calling forward twice in a row shouldn't change output
        y1 = m(input_)
        y2 = m(input_)
        assert torch.equal(y1, y2)

    def test_random_unstructured(self):
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "pruning.RandomPruningMethod.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            random_unstructured(m, name="weight", amount=0.9)

        y_postpruning = m(input_)
        y_postpruning.sum().backward()
        # weight_orig is the parameter, so it's the tensor that will accumulate the grad
        assert torch.equal(
            m.weight_orig.grad, mask
        )  # all 1s, except for masked units
        assert torch.equal(m.bias.grad, torch.ones_like(m.bias))

        # make sure that weight_orig update doesn't modify [1, 0] and [0, 3]
        old_weight_orig = m.weight_orig.clone()
        # update weights
        learning_rate = 1.0
        for p in m.parameters():
            p.data.sub_(p.grad.data * learning_rate)
        # since these are pruned, they should not be updated
        assert torch.equal(old_weight_orig[1, 0], m.weight_orig[1, 0])
        assert torch.equal(old_weight_orig[0, 3], m.weight_orig[0, 3])

    def test_random_unstructured_forward(self):
        """check forward with mask (by hand).
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.zeros_like(m.weight)
        mask[1, 0] = 1
        mask[0, 3] = 1

        with mock.patch(
            "pruning.RandomPruningMethod.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            random_unstructured(m, name="weight", amount=0.9)

        yhat = m(input_)
        assert torch.equal(yhat[0, 0], m.weight_orig[0, 3] + m.bias[0])
        assert torch.equal(yhat[0, 1], m.weight_orig[1, 0] + m.bias[1])

    def test_remove_pruning(self):
        """Remove pruning and check forward is unchanged from previous
        pruned state.
        """
        input_ = torch.ones(1, 5)
        m = nn.Linear(5, 2)

        # define custom mask to assign with mock
        mask = torch.ones_like(m.weight)
        mask[1, 0] = 0
        mask[0, 3] = 0

        # check grad is zero for masked weights
        with mock.patch(
            "pruning.RandomPruningMethod.compute_mask"
        ) as compute_mask:
            compute_mask.return_value = mask
            random_unstructured(m, name="weight", amount=0.9)

        y_postpruning = m(input_)

        remove(m, "weight")

        y_postremoval = m(input_)
        assert torch.equal(y_postpruning, y_postremoval)

    def test_tensor_id_consistency(self):
        m = nn.Linear(5, 2, bias=False)

        tensor_id = id(list(m.parameters())[0])

        random_unstructured(m, name="weight", amount=0.9)
        assert tensor_id == id(list(m.parameters())[0])

        remove(m, "weight")
        assert tensor_id == id(list(m.parameters())[0])

    @pytest.mark.parametrize("m", [nn.Linear(5, 7), nn.Conv3d(2, 2, 2)])
    @pytest.mark.parametrize("name", ["weight", "bias"])
    def test_random_unstructured_pickle(self, m, name):
        random_unstructured(m, name=name, amount=0.1)
        m_new = pickle.loads(pickle.dumps(m))
        assert isinstance(m_new, type(m))

    def test_multiple_pruning_calls(self):
        # if you call pruning twice, the hook becomes a PruningContainer
        m = nn.Conv3d(2, 2, 2)
        l1_unstructured(m, name="weight", amount=0.1)
        weight_mask0 = m.weight_mask  # save it for later sanity check

        # prune again
        ln_structured(m, name="weight", amount=0.3, n=2, axis=0)
        hook = next(iter(m._forward_pre_hooks.values()))
        assert isinstance(hook, PruningContainer)
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        assert hook._tensor_name == "weight"

        # check that the pruning container has the right length
        # equal to the number of pruning iters
        assert len(hook) == 2  # m.weight has been pruned twice

        # check that the entries of the pruning container are of the expected
        # type and in the expected order
        assert isinstance(hook[0], L1PruningMethod)
        assert isinstance(hook[1], LnStructuredPruningMethod)

        # check that all entries that are 0 in the 1st mask are 0 in the
        # 2nd mask too
        assert torch.all(m.weight_mask[weight_mask0 == 0] == 0)

        # prune again
        ln_structured(
            m, name="weight", amount=0.1, n=float("inf"), axis=1
        )
        # check that container._tensor_name is correctly set no matter how
        # many pruning methods are in the container
        hook = next(iter(m._forward_pre_hooks.values()))
        assert hook._tensor_name == "weight"

    def test_pruning_container(self):
        # create an empty container
        container = PruningContainer()
        container._tensor_name = "test"
        assert len(container) == 0

        p = L1PruningMethod(amount=2)
        p._tensor_name = "test"

        # test adding a pruning method to a container
        container.add_pruning_method(p)

        # test error raised if tensor name is different
        q = L1PruningMethod(amount=2)
        q._tensor_name = "another_test"
        with pytest.raises(ValueError):
            container.add_pruning_method(q)

        # test that adding a non-pruning method object to a pruning container
        # raises a TypeError
        with pytest.raises(TypeError):
            container.add_pruning_method(10)
        with pytest.raises(TypeError):
            container.add_pruning_method("ugh")

    def test_pruning_container_compute_mask(self):
        """Test `compute_mask` of pruning container with a known `t` and 
        `default_mask`. Indirectly checks that Ln structured pruning is 
        acting on the right axis.
        """
        # create an empty container
        container = PruningContainer()
        container._tensor_name = "test"

        # 1) test unstructured pruning
        # create a new pruning method
        p = L1PruningMethod(amount=2)
        p._tensor_name = "test"
        # add the pruning method to the container
        container.add_pruning_method(p)

        # create tensor to be pruned
        t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).to(dtype=torch.float32)
        # create prior mask by hand
        default_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 1]])
        # since we are pruning the two lowest magnitude units, the outcome of
        # the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 1, 0], [1, 1, 0, 1]]).to(
            dtype=t.dtype
        )
        computed_mask = container.compute_mask(t, default_mask).to(
            dtype=t.dtype
        )
        assert torch.equal(expected_mask, computed_mask)

        # 2) test structured pruning
        q = LnStructuredPruningMethod(amount=1, n=2, axis=0)
        q._tensor_name = "test"
        container.add_pruning_method(q)
        # since we are pruning the lowest magnitude one of the two rows, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 1]]).to(
            dtype=t.dtype
        )
        computed_mask = container.compute_mask(t, default_mask).to(
            dtype=t.dtype
        )
        assert torch.equal(expected_mask, computed_mask)

        # 3) test structured pruning, along another axis
        r = LnStructuredPruningMethod(amount=1, n=2, axis=1)
        r._tensor_name = "test"
        container.add_pruning_method(r)
        # since we are pruning the lowest magnitude of the four columns, the
        # outcome of the calculation should be this:
        expected_mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 1]]).to(
            dtype=t.dtype
        )
        computed_mask = container.compute_mask(t, default_mask).to(
            dtype=t.dtype
        )
        assert torch.equal(expected_mask, computed_mask)


    def test_l1_unstructured_pruning(self):
        """Test that l1 unstructured pruning actually removes the lowest 
        entries by l1 norm (by hand). It also checks that applying l1 
        unstructured pruning more than once respects the previous mask.
        """
        m = nn.Linear(4, 2)
        # modify its weight matrix by hand
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(
                dtype=torch.float32
            )
        )

        l1_unstructured(m, "weight", amount=2)
        expected_weight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]]).to(
            dtype=m.weight.dtype
        )
        assert torch.equal(expected_weight, m.weight)

        # check that pruning again removes the next two smallest entries
        l1_unstructured(m, "weight", amount=2)
        expected_weight = torch.tensor([[0, 0, 3, 4], [-4, -3, 0, 0]]).to(
            dtype=m.weight.dtype
        )
        assert torch.equal(expected_weight, m.weight)

    def test_unstructured_pruning_same_magnitude(self):
        """Since it may happen that the tensor to prune has entries with the
        same exact magnitude, it is important to check that pruning happens 
        consistenly based on the bottom % of weights, and not by threshold, 
        which would instead kill off *all* units with magnitude = threshold.
        """
        AMOUNT = 0.2
        p = L1PruningMethod(amount=AMOUNT)
        # create a random tensors with entries in {-2, 0, 2}
        t = 2 * torch.randint(low=-1, high=2, size=(10, 7))
        nparams_toprune = _compute_nparams_toprune(AMOUNT, t.nelement())

        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        nparams_pruned = torch.sum(computed_mask == 0)
        assert nparams_toprune == nparams_pruned

    def test_random_structured_pruning_amount(self):
        AMOUNT = 0.6
        AXIS = 2
        p = RandomStructuredPruningMethod(amount=AMOUNT, axis=AXIS)
        t = 2 * torch.randint(low=-1, high=2, size=(5, 4, 2)).to(
            dtype=torch.float32
        )
        nparams_toprune = _compute_nparams_toprune(AMOUNT, t.shape[AXIS])

        computed_mask = p.compute_mask(t, default_mask=torch.ones_like(t))
        # check that 1 column is fully prune, the others are left untouched
        remaining_axes = [_ for _ in range(len(t.shape)) if _ != AXIS]
        per_column_sums = sorted(
            torch.sum(computed_mask == 0, axis=remaining_axes
        ))
        assert per_column_sums == [0, 20]


    def test_global_pruning(self):
        """Test that global l1 unstructured pruning over 2 parameters removes
        the `amount=4` smallest global weights across the 2 parameters.
        """
        m = nn.Linear(4, 2)
        n = nn.Linear(3, 1)
        # modify the weight matrices by hand
        m.weight = torch.nn.Parameter(
            torch.tensor([[1, 2, 3, 4], [-4, -3, -2, -1]]).to(
                dtype=torch.float32)
        )
        n.weight = torch.nn.Parameter(
            torch.tensor([[0, 0.1, -2]]).to(
                dtype=torch.float32)
        )

        params_to_prune = (
            (m, 'weight'),
            (n, 'weight'),
        )

        # prune the 4 smallest weights globally by L1 magnitude
        global_unstructured(
            params_to_prune,
            pruning_method=L1PruningMethod,
            amount=4
        )

        expected_mweight = torch.tensor([[0, 2, 3, 4], [-4, -3, -2, 0]]).to(
            dtype=m.weight.dtype
        )
        assert torch.equal(expected_mweight, m.weight)

        expected_nweight = torch.tensor([[0, 0, -2]]).to(dtype=n.weight.dtype)
        assert torch.equal(expected_nweight, n.weight)


    def test_custom_from_mask_pruning(self):
        """Test that the CustomFromMaskPruningMethod is capable of receiving
        as input at instantiation time a custom mask, and combining it with
        the previous default mask to generate the correct final mask.
        """
        # new mask
        mask = torch.tensor([[0, 1, 1, 0], [0, 0, 1, 1]])
        # old mask
        default_mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])

         # some tensor (not actually used)
        t = torch.rand_like(mask.to(dtype=torch.float32))

        p = CustomFromMaskPruningMethod(mask=mask)

        computed_mask = p.compute_mask(t, default_mask).to(
            dtype=t.dtype
        )
        expected_mask = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]]).to(
            dtype=t.dtype
        )

        assert torch.equal(computed_mask, expected_mask)
