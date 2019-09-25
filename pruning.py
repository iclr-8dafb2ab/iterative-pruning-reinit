r"""
Pruning methods
"""
from abc import ABC, abstractmethod
from collections.abc import Iterable
import numpy as np
import torch

# TODO: if pruning fails, reinstate the original weight


class BasePruningMethod(ABC):
    def __init__(self):
        pass

    def __call__(self, module, inputs):
        """Multiplies the mask (stored in `module[name + '_mask']`)
        into the original tensor (stored in `module[name + '_orig']`)
        and stored the result into `module[name]` by using `apply_mask`.
        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t, default_mask):
        """Computes and returns a mask for the input tensor `t`.
        Starting from a base default_mask (which should be a mask of ones if the
        tensor has not been pruned yet), generate a random mask to apply on top of
        the default_mask according to specific pruning method recipe.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning iterations,
                that need to be respected after the new mask is applied.
                Same dims as `t`.
        Returns:
            mask (torch.Tensor): mask to apply to `t`, of same dims as `t`.
        """
        pass

    def apply_mask(self, module):
        """Simply handles the multiplication.
        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.
        Args:
            module (nn.Module): module containing the tensor to prune
        Returns:
            output (torch.Tensor): pruned tensor
        """
        # to carry out the multiplication, the mask needs to have been computed,
        # so the pruning method must know what tensor it's operating on
        assert (
            self._tensor_name is not None
        ), "Module {} has to be pruned".format(
            module
        )  # this gets set in apply()
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        output = mask.to(dtype=orig.dtype) * orig
        return output

    @classmethod
    def apply(cls, module, name, *args, **kwargs):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
            args: arguments passed on to a subclass of BasePruningMethod
            kwargs: keyword arguments passed on to a subclass of a BasePruningMethod
        """

        def _get_composite_method(cls, module, name, *args, **kwargs):
            # Check if a pruning method has already been applied to
            # `module[name]`. If so, store that in `old_method`.
            old_method = None
            found = 0
            # there should technically be only 1 hook with hook.name == name
            # assert this using `found`
            hooks_to_remove = []
            for k, hook in module._forward_pre_hooks.items():
                # if it exists, take existing thing, remove hook, then
                # go thru normal thing
                if (
                    isinstance(hook, BasePruningMethod)
                    and hook._tensor_name == name
                ):
                    old_method = hook
                    # # reset the tensor reparametrization
                    # module = remove_pruning(module, name)
                    hooks_to_remove.append(k)
                    found += 1
            assert (
                found <= 1
            ), "Avoid adding multiple pruning hooks to the\
                same tensor {} of module {}. Use a PruningContainer.".format(
                name, module
            )

            for k in hooks_to_remove:
                del module._forward_pre_hooks[k]

            # Apply the new pruning method, either from scratch or on top of
            # the previous one.
            method = cls(*args, **kwargs)  # new pruning
            # Have the pruning method remember what tensor it's been applied to
            setattr(method, "_tensor_name", name)

            # combine `methods` with `old_method`, if `old_method` exists
            if old_method is not None:  # meaning that there was a hook
                # if the hook is already a pruning container, just add the
                # new pruning method to the container
                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method  # rename old_method --> method

                # if the hook is simply a single pruning method, create a
                # container, add the old pruning method and the new one
                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)
                    # Have the pruning method remember the name of its tensor
                    # setattr(container, '_tensor_name', name)
                    container.add_pruning_method(method)
                    method = container  # rename container --> method
            return method

        method = _get_composite_method(cls, module, name, *args, **kwargs)
        # at this point we have no forward_pre_hooks but we could have an
        # active reparametrization of the tensor if another pruning method
        # had been applied (in which case `method` would be a PruningContainer
        # and not a simple pruning method).

        # Apply pruning to the module
        # original tensor, prior to this iteration of pruning (not like orig orig)
        orig = getattr(module, name)

        if not isinstance(method, PruningContainer):
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(name + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[name]
            default_mask = torch.ones_like(orig)  # temp

        else:
            default_mask = getattr(module, name + "_mask").detach().clone()

        # get the final mask, computed according to the specific method
        mask = method.compute_mask(orig, default_mask=default_mask)
        # reparametrize by saving mask to `module[name + '_mask']`...
        module.register_buffer(name + "_mask", mask)
        # ... and the new pruned tensor to `module[name]`
        setattr(module, name, method.apply_mask(module))

        # associate the pruning method to the module via a hook to
        # compute the function before every forward() (compile by run)
        module.register_forward_pre_hook(method)

        return method

    def remove(self, module):
        r"""Removes the pruning reparameterization from a module. The pruned
        parameter named `name` remains permanently pruned, and the parameter
        named `name+'_orig'` is removed from the parameter list. Similarly,
        the buffer named `name+'_mask' is removed from the buffers.

        Note:
            Pruning itself is NOT undone or reversed!
        """
        # before removing pruning from a tensor, it has to have been applied
        assert (
            self._tensor_name is not None
        ), "Module {} has to be pruned\
            before pruning can be removed".format(
            module
        )  # this gets set in apply()

        # to update module[name] to latest trained weights
        weight = self.apply_mask(module)  # masked weights

        # delete and reset
        delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + "_orig"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_orig"]
        del module._buffers[self._tensor_name + "_mask"]
        module.register_parameter(
            self._tensor_name, orig
        )


class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods.
    """

    def __init__(self, *args):
        self._pruning_methods = tuple()
        if not isinstance(args, Iterable):  # only 1 item
            setattr(self, "_tensor_name", args._tensor_name)
            self.add_pruning_method(args)
        elif len(args) == 1:  # only 1 item in a tuple
            setattr(self, "_tensor_name", args[0]._tensor_name)
            self.add_pruning_method(args[0])
        else:  # manual construction from list or other iterable
            for method in args:
                self.add_pruning_method(method)

    @classmethod
    def build_from(cls, obj):
        """Copy constructor
        Args:
            obj (PruningContainer): another PruningContainer object
        """
        if isinstance(obj, PruningContainer):
            # return copy.deepcopy(obj)
            return cls(obj.name, *obj._pruning_methods)
        else:
            raise TypeError("obj must be of type PruningContainer")

    def add_pruning_method(self, method):
        r"""Adds a child pruning method to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
        # check that we're adding a pruning method to the container
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(
                "{} is not a BasePruningMethod subclass".format(type(method))
            )
        elif self._tensor_name != method._tensor_name:
            raise ValueError(
                "Can only add pruning methods acting on "
                "the parameter named '{}' to PruningContainer {}.".format(
                    self._tensor_name, self
                )
                + " Found '{}'".format(method._tensor_name)
            )
        # if all checks passed, add to _pruning_methods tuple
        self._pruning_methods += (method,)

        # # fill attributes
        # self.cumulative_amounts += (self.get_cumulative_amount(), )

    def __len__(self):
        return len(self._pruning_methods)

    def __iter__(self):
        return iter(self._pruning_methods)

    # the following is inspired by torch.nn.modules.container
    def __getitem__(self, idx):
        """
        Args:
            idx (int or slice):
        Returns:
            a BasePruningMethod subclass
        """
        return self._pruning_methods[idx]

    def compute_mask(self, t, default_mask):
        def _combine_masks(method, t, mask):
            """Compute new cumulative mask from old mask * new partial mask.
            The new partial mask should be computed on the entries that
            were not zeroed out by the old mask.
            Which portions of the tensor the new mask will be calculated from
            depends on the PRUNING_TYPE (handled by the type handler):
            for 'unstructured', the mask will be computed from the raveled list
            of nonmasked entries;
            for 'structured', the mask will be computed from the nonmasked
            channels in the tensor.

            Args:
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor representing the parameter to prune
                    (of same dimensions as mask).
                mask (torch.Tensor): mask from previous pruning iteration
            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current
                    pruning method (of same dimensions as mask and t).
            """
            new_mask = mask  # start off from existing mask
            new_mask = new_mask.to(dtype=t.dtype)

            # compute a slice of t onto which the new pruning method will operate
            if method.PRUNING_TYPE == "unstructured":
                # prune entries of t where the mask is 1
                slc = mask == 1

            # for struct pruning, exclude channels that have already been
            # entirely pruned
            elif method.PRUNING_TYPE == "structured":
                if not hasattr(method, "axis"):
                    raise AttributeError(
                        "Pruning methods of PRUNING_TYPE "
                        '"structured" need to have the attribute `axis` defined.'
                    )

                # find the channels to keep by removing the ones that have been
                # zeroed out already (i.e. where sum(entries) == 0)
                n_dims = len(t.shape)  # "is this a 2D tensor? 3D? ..."
                axis = method.axis
                if axis == -1:
                    axis = n_dims - 1  # -1 means last dimension
                # find channels along dim = axis that aren't already tots 0ed out
                keep_channel = (
                    mask.sum(dim=[d for d in range(n_dims) if d != axis]) != 0
                )
                # create slice to identify what to prune
                slc = [slice(None)] * n_dims
                slc[axis] = keep_channel

            elif method.PRUNING_TYPE == "global":
                n_dims = len(t.shape)  # "is this a 2D tensor? 3D? ..."
                slc = [slice(None)] * n_dims

            else:
                raise ValueError(
                    "Unrecognized PRUNING_TYPE {}".format(method.PRUNING_TYPE)
                )

            # compute the new mask on the unpruned slice of the tensor t
            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)

            return new_mask

        # apply the latest method by combining the default mask with
        # the new mask that it generates
        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask


class IdentityPruningMethod(BasePruningMethod):
    """Doesn't prune any units.
    """

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        mask = default_mask
        return mask

    @classmethod
    def apply(cls, module, name):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
        """
        # this is here just for docstring generation for docs
        return super(IdentityPruningMethod, cls).apply(module, name)


class RandomPruningMethod(BasePruningMethod):
    """Prune units in a tensor at random.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        """
        Args:
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
        """
        # super(RandomPruningMethod, self).__init__()

        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()  # TODO: "size" is misleading
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone()

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # TODO: torch.sparse?
            # Create random mask with nparams_nparams_toprune entries set to 0
            prob = torch.rand_like(t)
            topk = torch.topk(
                prob.view(-1), k=nparams_toprune,
            )
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
        """
        # this is here just for docstring generation for docs
        return super(RandomPruningMethod, cls).apply(
            module, name, amount=amount
        )


class L1PruningMethod(BasePruningMethod):
    """Prune units in a tensor by zeroing out the ones with the lowest L1-norm.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        """
        Args:
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
        """
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()  # TODO: "size" is misleading
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone()

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Keep the smallest k
            topk = torch.topk(
                torch.abs(t).view(-1), k=nparams_toprune, largest=False
            )
            # topk will have .indices and .values

            # Compute binary mask by setting the entries corresponding to the
            # bottom k values equal to 0
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
        """
        # this is here just for docstring generation for docs
        return super(L1PruningMethod, cls).apply(module, name, amount=amount)


# TODO: decide what to do about structured pruning on 1D tensors like bias.
#       Prune the entire tensor, or raise an error? Raising error for now.
class RandomStructuredPruningMethod(BasePruningMethod):
    """Prune entire channels in a tensor at random.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, axis=-1):
        """
        Args:
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
            axis (int, optional): index of the axis along which we define
                channels to prune. Default: -1.
        """
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.axis = axis

    def compute_mask(self, t, default_mask):
        """Computes and returns a mask for the input tensor `t`.
        Starting from a base default_mask (which should be a mask of ones if the
        tensor has not been pruned yet), generate a random mask to apply on top of
        the default_mask by randomly zeroing out channels along the specified
        axis of the tensor.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning iterations,
                that need to be respected after the new mask is applied.
                Same dims as `t`.
        Returns:
            mask (torch.Tensor): mask to apply to `t`, of same dims as `t`.
        Raises:
            IndexError: if self.axis >= len(t.shape)
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)

        # Check that self.axis is a valid axis to index t, else raise IndexError
        _validate_pruning_axis(t, self.axis)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the axis to prune
        tensor_size = t.shape[self.axis]  # TODO: "size" is misleading
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.axis.
        # mask has the same shape as tensor t
        def make_mask(t, axis, nchannels, nchannels_toprune):
            # generate a random number in [0, 1] to associate to each channel
            prob = torch.rand(nchannels)
            # generate mask for each channel by 0ing out the channels that
            # got assigned the k = nchannels_toprune highest values in prob
            topk = torch.topk(
                prob, k=nchannels_toprune,
            )
            channel_mask = torch.ones(size=(nchannels, ))
            channel_mask.view(-1)[topk.indices] = 0
            channel_mask = channel_mask.to(dtype=torch.uint8)

            # use the channel mask to broadcast the masking up to the 
            # dimension of t
            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[axis] = channel_mask
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            # mask = torch.ones_like(t)
            mask = default_mask
        else:
            # apply the new structured mask on top of prior (potentially unstructured) mask
            mask = make_mask(t, self.axis, tensor_size, nparams_toprune)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, amount, axis=-1):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
        axis (int, optional): index of the axis along which we define channels
                to prune. Default: -1.
        """
        # this is here just for docstring generation for docs
        return super(RandomStructuredPruningMethod, cls).apply(
            module, name, amount=amount, axis=axis
        )


class LnStructuredPruningMethod(BasePruningMethod):
    """Prune entire channels in a tensor based on their Ln-norm.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, n, axis=-1):
        """
        Args:
            amount (int or float): quantity of channels to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument p in torch.norm
            axis (int, optional): index of the axis along which we define
                channels to prune. Default: -1.
        """
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.axis = axis

    def compute_mask(self, t, default_mask):
        """Computes and returns a mask for the input tensor `t`.
        Starting from a base default_mask (which should be a mask of ones if the
        tensor has not been pruned yet), generate a mask to apply on top of
        the default_mask by zeroing out the channels along the specified
        axis with the lowest Ln-norm.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning iterations,
                that need to be respected after the new mask is applied.
                Same dims as `t`.
        Returns:
            mask (torch.Tensor): mask to apply to `t`, of same dims as `t`.
        Raises:
            IndexError: if self.axis >= len(t.shape)
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        # Check that self.axis is a valid axis to index t, else raise IndexError
        _validate_pruning_axis(t, self.axis)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the axis to prune
        tensor_size = t.shape[self.axis]  # TODO: "size" is misleading
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # L_n norm along each channel to then find the topk based on this
        # metric
        norm = _compute_norm(t, self.n, self.axis)
        # largest=True --> top k; largest=False --> bottom k
        # Keep the largest k channels along dim=self.axis
        topk = torch.topk(
            norm,
            k=nparams_tokeep,
            # dim=self.axis,
            largest=True,
        )
        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.axis.
        # mask has the same shape as tensor t
        def make_mask(t, axis, indices):
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=axis with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if axis=2 & indices=[0,2,3]
            slc[axis] = indices
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            # mask = torch.ones_like(t)
            mask = default_mask
        else:
            mask = make_mask(t, self.axis, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        # TODO: it might make more sense to set largest=False to find bottom k,
        # then set mask to 1s everywhere, except where bottomk.indices says we
        # should fill in zeros.
        return mask

    @classmethod
    def apply(cls, module, name, amount, n, axis):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument p in torch.norm
            axis (int): index of the axis along which we define channels to prune.
        """
        # this is here just for docstring generation for docs
        return super(LnStructuredPruningMethod, cls).apply(
            module, name, amount=amount, n=n, axis=axis
        )

class CustomFromMaskPruningMethod(BasePruningMethod):

    PRUNING_TYPE = "global"

    def __init__(self, mask):
        self.mask = mask

    def compute_mask(self, t, default_mask):
        assert default_mask.shape == self.mask.shape
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, mask):
        """Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (string): parameter name within `module` on which pruning
                will act.
        """
        # this is here just for docstring generation for docs
        return super(CustomFromMaskPruningMethod, cls).apply(module, name, mask)


def identity(module, name):
    """Applies pruning reparametrization to the tensor corresponding to the
    parameter called `name` in `module` without actually pruning any units.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Note:
        The mask is a tensor of ones.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module,
    """
    IdentityPruningMethod.apply(module, name)
    return module


def random_unstructured(module, name, amount):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing the specified `amount` of units selected at random.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.
    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module,
    """
    RandomPruningMethod.apply(module, name, amount)
    return module


def l1_unstructured(module, name, amount):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing the specified `amount` of units with the lowest L1-norm.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.
    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module,
    """
    L1PruningMethod.apply(module, name, amount)
    return module


def random_structured(module, name, amount, axis):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing the specified `amount` of channels along the specified
    `axis` selected at random.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        axis (int): index of the axis along which we define channels to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module
    """
    RandomStructuredPruningMethod.apply(module, name, amount, axis)
    return module


def ln_structured(module, name, amount, n, axis):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing the specified `amount` of channels along the specified
    `axis` with the lowest L`n`-norm.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        axis (int): index of the axis along which we define channels to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module,
    """
    LnStructuredPruningMethod.apply(module, name, amount, n, axis)
    return module

def global_unstructured(parameters, pruning_method, **kwargs):
    """
    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type nn.Module, and name must be a string.
        pruning_method (function): a valid pruning function from this module, 
            or a custom one implemented by the user that satisfies the 
            implementation guidelines and has PRUNING_TYPE='unstructured'.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the 
                specified parameters.
                If float, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If int, it represents the 
                absolute number of parameters to prune.

    Raises:
        TypeError: if PRUNING_TYPE != 'unstructured'

    Note:
        Since global structured pruning doesn't make much sense unless the 
        norm is normalized by the size of the parameter, we now limit the 
        scope of global pruning to unstructured methods.
    """
    # ensure parameters is a list or generator of tuples
    assert isinstance(parameters, Iterable)

    # flatten parameter values to consider them all at once in global pruning
    t = torch.nn.utils.parameters_to_vector([getattr(*p) for p in parameters])
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector([
        getattr(module, name + '_mask', torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
    ])

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = PruningContainer()
    container._tensor_name = 'temp'  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = 'temp' # to make it match that of `container`
    if method.PRUNING_TYPE != 'unstructured':
        raise TypeError('Only "unstructured" PRUNING_TYPE supported for '
            'the `pruning_method`. Found method {} of type {}'.format(
                pruning_method, method.PRUNING_TYPE))

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the 
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(t, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:

        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer:pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it 
        # to the forward_pre_hooks like any other pruning method
        custom_from_mask(module, name, param_mask)

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param
        

def custom_from_mask(module, name, mask):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by applying the pre-computed mask in `mask`.
    Modifies module in place (and also return the modified module) 
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the 
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the 
    original (unpruned) parameter is stored in a new parameter named 
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module, 
    """
    CustomFromMaskPruningMethod.apply(module, name, mask)
    return module


def remove(module, name):
    r"""Removes the pruning reparameterization from a module and the
    pruning method from the forward hook. The pruned
    parameter named `name` remains permanently pruned, and the parameter
    named `name+'_orig'` is removed from the parameter list. Similarly,
    the buffer named `name+'_mask' is removed from the buffers.

    Note:
        Pruning itself is NOT undone or reversed!

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Example:
        >>> m = random_pruning(nn.Linear(5, 7), name='weight', amount=0.2)
        >>> remove_pruning(m, name='weight')
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(
        "Parameter '{}' of module {} has to be pruned "
        "before pruning can be removed".format(name, module)
    )

def is_pruned(model):
    """Check whether `model` is pruned by looking for forward_pre_hooks in its
    modules that inherit from the BasePruningMethod.
    Args:
        model (torch.nn.Module): object that is either pruned or unpruned
    Returns:
        binary answer to whether `model` is pruned.
    """
    for _, module in model.named_module():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, BasePruningMethod):
                return True
    return False

def _validate_pruning_amount_init(amount):
    """Validation helper to check the range of amount at init.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
                    integer.
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
        Inspired by scikit-learn train_test_split.
    """
    amount_type = np.asarray(amount).dtype.kind

    if (
        amount_type == "f"
        and (amount > 1.0 or amount < 0.0)
        or amount_type == "i"
        and amount < 0
    ):
        raise ValueError(
            "amount={} should either be a float in the "
            "range [0, 1] or a non-negative integer"
            "".format(amount)
        )

    if amount_type not in ("i", "f"):
        raise TypeError(
            "Invalid type for amount: {}. Must be int or float."
            "".format(amount)
        )


def _validate_pruning_amount(amount, tensor_size):
    """Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (tensor_size).

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Note:
        Inspired by scikit-learn train_test_split.
    """
    # TODO: consider removing this check and allowing users to specify
    # a number of units to prune that is greater than the number of units
    # left to prune. In this case, the tensor will just be fully pruned.
    amount_type = np.asarray(amount).dtype.kind

    if amount_type == "i" and amount > tensor_size:
        raise ValueError(
            "amount={} should be smaller than the number of "
            "parameters to prune={}".format(amount, tensor_size)
        )


def _validate_structured_pruning(t):
    """Validation helper to check that the tensor to be pruned is multi-
    dimensional, such that the concept of "channels" is well-defined.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune

    Raises:
        ValueError: if the tensor t is not at least 2D.
    """
    shape = t.shape
    if len(shape) <= 1:
        raise ValueError(
            "Structured pruning can only be applied to "
            "multidimensional tensors. Found tensor of shape "
            "{} with {} dims".format(shape, len(shape))
        )


def _compute_nparams_toprune(amount, tensor_size):
    """Since amount can be expressed either in absolute value or as a
    percentage of the number of units/channels in a tensor, this utility
    function converts the percentage to absolute value to standardize
    the handling of pruning.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Returns:
        int: the number of units to prune in the tensor
    """
    amount_type = np.asarray(amount).dtype.kind

    if amount_type == "i":
        return amount
    elif amount_type == "f":
        return round(amount * tensor_size)
    # incorrect type already checked in _validate_pruning_amount_init


def _validate_pruning_axis(t, axis):
    """
    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        axis (int): index of the axis along which we define channels to prune
    """
    if axis >= len(t.shape):
        raise IndexError(
            "Invalid index {} for tensor of size {}".format(axis, t.shape)
        )


def _compute_norm(t, n, axis):
    """Compute the L_n-norm across all entries in tensor t along all dimension
    except for the one identified by axis.
    Example: if t is of shape, say, 3x2x4 and axis=2 (the last axis),
    then norm will have Size [4], and each entry will represent the
    L_n-norm computed using the 3x2=6 entries for each of the 4 channels.
    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        axis (int): Axis identifying the channels to prune
    Returns:
        norm (torch.Tensor): L_n norm computed across all dimensions except
            for `axis`. norm.shape = t.shape[-1]
    """
    # dims = all axes, except for the one identified by `axis`
    dims = list(range(len(t.shape)))
    if axis == -1:
        axis = dims[-1]
    dims.remove(axis)

    norm = torch.norm(t, p=n, dim=dims)
    return norm
