import unittest
import autoray as ar
from autoray import numpy as np
import numpy
from semiring_torch import log_torch


def test_zeros():
    return np.zeros((1, 2, 2))


def test_ones():
    return np.ones((1, 2, 2))


def test_tensor():
    return np.array([[1, 2], [5, 6]])


def test_add_constant():
    return test_zeros() + 0.5


def test_mul_constant():
    return 0.5 * test_ones()


def test_add():
    return test_tensor() + test_tensor()


def test_mul():
    return test_tensor() * test_tensor()


def test_t():
    return test_tensor().transpose(1, 0)


def test_squeeze():
    return test_ones().squeeze()


def test_flatten():
    return test_ones().flatten()


def test_sum():
    return test_tensor().sum()


def test_prod():
    return test_tensor().prod()


def test_matmul():
    return test_tensor() @ test_tensor()


def test_trace():
    return np.trace(test_tensor())


def test_einsum():
    return np.einsum('ij,jk->ik', test_tensor(), np.ones((2, 2)))


def test_dot():
    return np.dot(np.array([1, 2]), np.array([3, 4]))


def test_outer():
    return np.outer(np.array([1, 2]), np.array([3, 4]))


def test_indexing():
    return test_tensor()[0, 0]


def test_assignment():
    x = test_tensor()
    x[0] = 5
    return x


TESTS = [
    test_zeros,
    test_ones,
    test_tensor,
    test_add_constant,
    test_mul_constant,
    test_add,
    test_mul,
    test_t,
    test_squeeze,
    test_flatten,
    test_sum,
    test_prod,
    test_matmul,
    test_trace,
    test_einsum,
    test_dot,
    test_outer,
    test_indexing,
    test_assignment,
]


class LogSemiringTests(unittest.TestCase):
    backends = ['numpy', 'log_torch']

    def tester(self):
        for func in TESTS:
            print("### Testing", func.__name__)
            expected_result = None
            for backend in self.backends:
                with ar.backend_like(backend):
                    backend_result = ar.to_numpy(func())
                    print('backend', backend_result)
                    if expected_result is None:
                        expected_result = backend_result
                    else:
                        numpy.testing.assert_allclose(expected_result, backend_result, rtol=1e-5)

