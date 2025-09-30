from collections.abc import Callable

import numpy as np
import pytest

from mno.function import Function
from mno.my_types import Vec

functions_with_derivatives = [
    [
        lambda point: point[0] ** 2 - point[1],
        lambda point: np.array([point[0] * 2, -1]),
    ],
    [
        lambda p: p[0] / p[1] - p[2] ** p[3],
        lambda p: np.array(
            [
                1 / p[1],
                -p[0] / (p[1] ** 2),
                -p[3] * p[2] ** (p[3] - 1),
                p[2] ** p[3] * np.log(p[2]),
            ]
        ),
    ],
]


@pytest.mark.parametrize(
    ("function", "point"),
    [
        (functions_with_derivatives[0][0], np.array([1, 2, 3])),
        (functions_with_derivatives[1][0], np.array([0, 0.1, 1, 2])),
    ],
)
def test_function_without_derivative_get(function: Callable[[Vec], float], point: Vec):
    assert Function(function)(point) == function(point)


@pytest.mark.parametrize(
    ("function", "derivative", "point"),
    [
        (*functions_with_derivatives[0], np.array([1, 2, 3])),
        (*functions_with_derivatives[1], np.array([0, 0.1, 1, 2])),
    ],
)
def test_function_with_derivative_get(
    function: Callable[[Vec], float], derivative: Callable[[Vec], Vec], point: Vec
):
    assert Function(function, derivative)(point) == function(point)


def test_numerical_grad_of_constant_is_zero():
    f: Function = Function(lambda _: 2)
    df = f.numerical_grad()
    for dim in range(1, 5):
        for vec in np.ndindex((10,) * dim):
            assert all(df(np.array(vec) - 5) < 1e-9)


def test_numerical_graf_of_grad_of_constant_is_zero():
    f: Function = Function(lambda _: 2)
    df = f.numerical_grad().numerical_grad()
    for dim in range(1, 5):
        for vec in np.ndindex((10,) * dim):
            print(df(np.array(vec) - 5))
            assert all(df(np.array(vec) - 5).reshape(-1) < 1e-9)
