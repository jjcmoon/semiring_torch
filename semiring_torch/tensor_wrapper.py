import abc
from inspect import signature

import torch
import numpy as np


WRAPPED_METHODS = {
    # meta
    'numel',
    'nelement',
    'dim',
    # 'as_strided',
    'backward',
    # 'bfloat16',
    'cpu',
    'cuda',
    'contiguous',
    # 'data_ptr',
    # 'dense_dim',
    # 'detach',
    # 'double',
    # 'float',
    'get_device',
    # 'half',
    # 'indices',
    # 'long',
    'pin_memory',
    # 'short',

    # Tensor shape manipulation
    'broadcast_to',
    # 'chunk',
    'clone',
    'copy_',
    'diagonal',
    'diagonal_scatter',
    # 'dsplit',
    'expand',
    # 'expand_as',
    'flatten',
    'flip',
    'fliplr',
    'flipud',
    'gather',
    # 'hsplit',
    'movedim',
    'moveaxis',
    'permute',
    'ravel',
    'repeat',
    'roll',
    'rot90',
    'select',
    # 'split',
    'squeeze',
    'squeeze_',
    'swapaxes',
    'swapdims',
    't',
    'transpose',
}

WRAPPED_ATTRIBUTES = {
    'is_cuda',
    'is_quantized',
    'is_contiguous',
    'is_floating_point',
    'is_leaf',
    'is_pinned',
    'is_set_to',
    'is_shared',
    'is_sparse',

    'ndim',
    'nbytes',
    'shape',
    'device',
    'dtype',
    'requires_grad',
}


class TensorWrapper(abc.ABC):
    def __init__(self, data):
        assert data.dtype in self.ADMISSIBLE_DTYPES, \
            f"Invalid dtype: {data.dtype} for {self.__class__.__name__}"
        self.data = data

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __str__(self):
        return str(self.outof_this(self.data))

    def __repr__(self):
        return repr(self.data)

    def __matmul__(self, other):
        return self._einsum("ij,jk->ik", self, other)

    def trace(self):
        return self.diagonal().sum()

    def dot(self, other):
        return (self * other).sum()

    def outer(self, other):
        return self[:, None] * other[None, :]

    def __getitem__(self, item):
        return self.__class__(self.data[item])

    def __setitem__(self, key, value):
        self.data[key] = self.convert_into(value).data

    def __getattr__(self, item):
        if item in WRAPPED_ATTRIBUTES:
            return self.data.__getattribute__(item)
        else:
            return self.__getattribute__(item)

    def __setattr__(self, key, value):
        if key in WRAPPED_ATTRIBUTES:
            self.data.__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    @classmethod
    def into_this(cls, tensor: torch.Tensor) -> "TensorWrapper":
        return TensorWrapper(tensor)

    @classmethod
    def outof_this(cls, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def torch(self) -> torch.Tensor:
        return self.outof_this(self.data)

    def numpy(self) -> np.ndarray:
        return self.torch().detach().cpu().numpy()

    @classmethod
    def convert_into(cls, value):
        if isinstance(value, cls):
            return value
        else:
            value = cls.into_this(torch.as_tensor(value))
            return cls(value)


def inject_tensor_methods(cls, methods):
    for method_name in methods:
        method = getattr(torch.Tensor, method_name)
        sig = _get_signature(method)
        if ": Tensor" in sig:
            raise NotImplementedError(f"Can't handle methods with Tensor arguments yet: {method_name}")

        patched_method = create_wrapper_method(method, sig)
        setattr(cls, method_name, patched_method)


def create_wrapper_method(method, sig):
    if sig.endswith('-> Tensor'):
        def patched_method(self, *args, **kwargs):
            return self.__class__(method(self.data, *args, **kwargs))
    else:
        def patched_method(self, *args, **kwargs):
            return method(self.data, *args, **kwargs)
    return patched_method


def _get_signature(method):
    try:
        sig = str(signature(method))
        sig = sig.replace('self, ', '')
    except ValueError:
        # extract signature from docstring
        doc = method.__doc__.strip()
        sig = doc.split('\n')[0].strip()
        assert '(' in sig and ')' in sig, f"Couldn't parse signature from {doc}"
        sig = sig[sig.index('('):]
    sig = sig.replace('other', 'other: Tensor')
    return sig


inject_tensor_methods(TensorWrapper, WRAPPED_METHODS)


