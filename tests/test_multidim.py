import pytest

from mno.find_distance import GoldsteinTest
from mno.function import Function
from mno.linesearch import GoldenSearch
from mno.multidimensional import (
    BFGSMethod,
    BroydenMethod,
    ConjugateGradient,
    DFPMethod,
    GradientDescend,
    MultidimensionalOptimalization,
)
from mno.my_types import Vec, array_to_vec, dot_prorduct

methods = [GradientDescend, ConjugateGradient, DFPMethod, BFGSMethod, BroydenMethod]


def add_methods_into_parametrization(*parametrization: tuple) -> list[tuple]:
    """Replicates parametrization for each method."""
    return [(method, *x) for x in parametrization for method in methods]


par = [(1, 2, 0, -2)]
PRECISION = 1e-3


def check_result(result: Vec, *expected: Vec) -> None:
    """Check relative error is less than required precision."""
    assert any(
        all(abs((result - e) / e) < PRECISION)
        or all(abs((e - result) / result) < PRECISION)
        or all(abs(e - result) < PRECISION)
        for e in expected
    )


@pytest.mark.parametrize(
    ("method", "c", "peak", "radius", "point"),
    add_methods_into_parametrization(
        (1, [0, 3], [2, 5], [3, 5]),
        (3, [-2, 5, 8, -2, 3], [3, 8, 1, 0.1, 1], [1, 2, 3, 4, 5]),
        (-3, [-2, 5, 8, -2, 3], [3, 8, 1, 0.1, 1], [1, 2, 3, 4, 5]),
    ),
)
def test_multidim_finds_solution_on_multi_dim_quadratic_function(
    method: type[MultidimensionalOptimalization],
    c: float,
    peak: list[float],
    radius: list[float],
    point: list[float],
) -> None:
    """Check if linesearch properly works on multi dimensional functions."""
    assert all(r > 0 for r in radius), "Invalid dataset, k need to be positive!"
    f = Function(
        lambda x: dot_prorduct(array_to_vec(radius), (x - array_to_vec(peak)) ** 2) + c,
        dim_in=len(radius),
        dim_out=1,
    )
    result = (
        method()
        .set_function(f)
        .set_point(array_to_vec(point))
        .set_distance_finder(GoldsteinTest())
        .set_linesearch(GoldenSearch())
        # .set_stopping_condition(IterationCondition())
        .solve()
    )
    check_result(result, array_to_vec(peak))


@pytest.mark.parametrize(
    ("method", "function", "derivative", "point"),
    add_methods_into_parametrization(
        (
            lambda a: (1.5 - a[0] + a[0] * a[1]) ** 2
            + (2.25 - a[0] + a[0] * a[1] ** 2) ** 2
            + (2.625 - a[0] + a[0] * a[1] ** 3) ** 2,
            lambda a: array_to_vec(
                [
                    2 * (1.5 - a[0] + a[0] * a[1]) * (a[1] - 1)
                    + 2 * (2.25 - a[0] + a[0] * a[1] ** 2) * (a[1] ** 2 - 1)
                    + 2 * (2.625 - a[0] + a[0] * a[1] ** 3) * (a[1] ** 3 - 1),
                    (1.5 - a[0] + a[0] * a[1]) * 2 * a[0]
                    + (2.25 - a[0] + a[0] * a[1] ** 2) * 4 * a[0] * a[1]
                    + (2.625 - a[0] + a[0] * a[1] ** 3) * 6 * a[0] * a[1] ** 2,
                ]
            ),
            [0, 0],
        )
    ),
)
def test_multidim_finds_solution_on_multi_dim_arbitrary_function(
    method: type[MultidimensionalOptimalization],
    function: callable,
    derivative: callable,
    point: list[Vec],
) -> None:
    """Check if linesearch properly works on multi dimensional functions."""
    f = Function(
        function=function,
        derivative=derivative,
        dim_in=len(point),
        dim_out=1,
    )
    result = (
        method()
        .set_function(f)
        .set_point(array_to_vec(point))
        .set_distance_finder(GoldsteinTest())
        .set_linesearch(GoldenSearch())
        # .set_stopping_condition(IterationCondition())
        .solve()
    )

    assert all(abs(f.grad()(result)) < PRECISION)
