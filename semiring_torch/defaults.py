import sys

import autoray
import numpy as np

from semiring_torch.tensor_wrapper import TensorWrapper


def array(data):
    # bit hacky...
    cls = autoray.numpy.zeros((0,)).__class__
    return cls.convert_into(data)


def to_numpy(tensor: TensorWrapper) -> np.ndarray:
    return tensor.numpy()


def eye(*args, **kwargs):
    return array(np.eye(*args, **kwargs))


def trace(tensor: TensorWrapper):
    return tensor.trace()


def dot(input, other):
    return input.dot(other)


def outer(input, other):
    return input.outer(other)


DEFAULT_FUNCS = {
    'array': array,
    'tensor': array,  # alias for 'array'
    'to_numpy': to_numpy,
    'eye': eye,
    'trace': trace,
    'dot': dot,
    'outer': outer,
}


def add_default_funcs(module):
    module = sys.modules[module]
    for name, func in DEFAULT_FUNCS.items():
        # print('setting', module, name, func)
        setattr(module, name, func)
