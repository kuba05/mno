from collections.abc import Callable

import numpy as np
import pytest

from mno.function import Function
from mno.my_types import Vec, array_to_vec, float_to_vec

functions_with_derivatives = [
    [
        lambda point: float_to_vec(point[0] ** 2 - point[1]),
        lambda point: array_to_vec([point[0] * 2, -1, 0]),
        (3, 1),
    ],
    [
        lambda p: float_to_vec(p[0] / p[1] - p[2] ** p[3]),
        lambda p: array_to_vec(
            [
                1 / p[1],
                -p[0] / (p[1] ** 2),
                -p[3] * p[2] ** (p[3] - 1),
                p[2] ** p[3] * np.log(p[2]),
            ]
        ),
        (4, 1),
    ],
]


@pytest.mark.parametrize(
    ("function", "dims", "point"),
    [
        (
            functions_with_derivatives[0][0],
            functions_with_derivatives[0][2],
            np.array([1, 2, 3]),
        ),
        (
            functions_with_derivatives[1][0],
            functions_with_derivatives[1][2],
            np.array([0, 0.1, 1, 2]),
        ),
    ],
)
def test_function_without_derivative_get(
    function: Callable[[Vec], Vec], dims: tuple[int, int], point: Vec
):
    """Test if a function returns correct value."""
    assert Function(function, dim_in=dims[0], dim_out=dims[1])(point) == function(point)


@pytest.mark.parametrize(
    ("function", "derivative", "dims", "point"),
    [
        (*functions_with_derivatives[0], np.array([1, 2, 3])),
        (*functions_with_derivatives[1], np.array([0, 0.1, 1, 2])),
    ],
)
def test_function_with_derivative_get_derivative(
    function: Callable[[Vec], Vec],
    derivative: Callable[[Vec], Vec],
    dims: tuple[int, int],
    point: Vec,
):
    assert all(
        Function(
            function, derivative=derivative, dim_in=dims[0], dim_out=dims[1]
        ).grad()(point)
        == derivative(point)
    )


@pytest.mark.parametrize(
    ("function", "derivative", "dims", "point"),
    [
        (*functions_with_derivatives[0], np.array([1, 2, 3])),
        (*functions_with_derivatives[1], np.array([0, 0.1, 1, 2])),
    ],
)
def test_function_with_derivative_get(
    function: Callable[[Vec], Vec],
    derivative: Callable[[Vec], Vec],
    dims: tuple[int, int],
    point: Vec,
):
    assert Function(function, derivative=derivative, dim_in=dims[0], dim_out=dims[1])(
        point
    ) == function(point)


def test_numerical_grad_of_constant_is_zero():
    for dim in range(1, 5):
        f: Function = Function(lambda _: float_to_vec(2), dim_in=dim, dim_out=1)
        df = f.numerical_grad()
        for vec in np.ndindex((10,) * dim):
            assert all(df(np.array(vec) - 5) < 1e-9)


def test_numerical_graf_of_grad_of_constant_is_zero():
    for dim in range(1, 5):
        f: Function = Function(lambda _: float_to_vec(2), dim_in=dim, dim_out=1)
        df = f.numerical_grad().numerical_grad()
        for vec in np.ndindex((6,) * dim):
            assert all(df(np.array(vec) - 3) < 1e-9)


def test_numerical_graf_of_linear():
    for dim in range(1, 4):
        for i in range(dim):
            f: Function = Function(lambda a: a[i], dim_in=dim, dim_out=1)
            df = f.numerical_grad()
            exp = np.array([0] * dim)
            exp[i] = 1
            for vec in np.ndindex((10,) * dim):
                assert all(abs(df(np.array(vec) - 5) - exp) < 1e-9)
