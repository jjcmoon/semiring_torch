import torch
import autoray

from semiring_torch.torch_semiring_einsum import compile_equation, log_einsum
from semiring_torch.tensor_wrapper import TensorWrapper
from semiring_torch.defaults import add_default_funcs


add_default_funcs(__name__)


class LogTensor(TensorWrapper):
    ADMISSIBLE_DTYPES = {torch.float32, torch.float64}

    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def _einsum(self, *args):
        return einsum(*args)

    def prod(self, *args, **kwargs):
        return prod(self, *args, **kwargs)

    def sum(self, *args, **kwargs):
        return sum(self, *args, **kwargs)

    @classmethod
    def into_this(cls, tensor: torch.Tensor):
        return tensor.log()

    @classmethod
    def outof_this(cls, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.exp()


def add(self, other):
    self = array(self).data
    other = array(other).data
    return LogTensor(torch.logaddexp(self, other))


def mul(self, other):
    self = array(self).data
    other = array(other).data
    return LogTensor(self + other)


def sum(self, **kwargs):
    self = array(self).data
    if 'dim' not in kwargs:
        kwargs['dim'] = tuple(range(len(self.shape)))
    return LogTensor(torch.logsumexp(self.data, **kwargs))


def prod(self, **kwargs):
    self = array(self).data
    return LogTensor(torch.sum(self.data, **kwargs))


def ones(*args, **kwargs):
    return LogTensor(torch.zeros(*args, **kwargs))


def zeros(*args, **kwargs):
    return LogTensor(torch.full(*args, fill_value=float("-inf"), **kwargs))


def einsum(*args):
    eq = compile_equation(args[0])
    args = [array(arg).data for arg in args[1:]]
    return LogTensor(log_einsum(eq, *args))


autoray.register_backend(LogTensor, "log_torch")
autoray.autoray._MODULE_ALIASES["log_torch"] = 'semiring_torch.log_torch'
