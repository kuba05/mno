import pytest

from mno.function import Function
from mno.linesearch import GoldenSearch, Linesearch, TernarySearch
from mno.my_types import array_to_vec, float_to_vec, dot_prorduct

methods = [TernarySearch, GoldenSearch]


def add_methods_into_parametrization(*parametrization: tuple) -> list[tuple]:
    """Replicates parametrization for each method."""
    return [(method, *x) for x in parametrization for method in methods]


par = [(1, 2, 0, -2)]


@pytest.mark.parametrize(
    ("method", "k", "c", "x_0", "x_1"),
    add_methods_into_parametrization(
        (1, 2, 0, -2),
        (3, -3, -2, -3),
        (0.0001, -1000, -10000, 1000),
        (1e-9, 1e9, -1e9, 1e9),
        (1, -3, 1, 1),
    ),
)
def test_linesearch_improves_solution_on_quadratic_function(
    method: type[Linesearch], k: float, c: float, x_0: float, x_1: float
) -> None:
    """Linesearch should easily find minimum of quadratic function."""
    assert k > 0, "Invalid dataset, k need to be positive!"
    f = Function(lambda x: (x - x_0) * (x - x_1) + c, dim_in=1, dim_out=1)
    sol = float_to_vec((x_0 + x_1) / 2)
    assert (
        method()
        .set_function(f)
        .set_interval(float_to_vec(x_0), float_to_vec(x_1))
        .solve()
        - sol
        < 1e-9
    )


@pytest.mark.parametrize(
    ("method", "c", "peak", "radius", "direction"),
    add_methods_into_parametrization(
        (1, [0, 3], [2, 5], [3, 5]),
        (3, [-2, 5, 8, -2, 3], [3, 8, 1, 0.1, 1], [1, 2, 3, 4, 5]),
        (-3, [-2, 5, 8, -2, 3], [3, 8, 1, 0.1, 1], [1, 2, 3, 4, 5]),
        (-1e9, [1e8, 1e7, 1e6, 1e10, 1e5], [1, 2, 3, 4, 5], [0, 0, 0, 0, 1e5]),
    ),
)
def test_linesearch_improves_solution_on_multi_dim_quadratic_function(
    method: type[Linesearch],
    c: float,
    peak: list[float],
    radius: list[float],
    direction: list[float],
) -> None:
    """Check if linesearch properly works on multi dimensional functions."""
    assert all(r > 0 for r in radius), "Invalid dataset, k need to be positive!"
    f = Function(
        lambda x: dot_prorduct(array_to_vec(radius), (x - array_to_vec(peak)) ** 2) + c,
        dim_in=len(radius),
        dim_out=1,
    )
    assert all(
        method()
        .set_function(f)
        .set_interval(
            array_to_vec(peak) - array_to_vec(direction),
            array_to_vec(peak) + array_to_vec(direction),
        )
        .solve()
        - peak
        < 1e-9
    )


@pytest.mark.parametrize(
    ("method", "k", "c", "x_0", "x_1"),
    add_methods_into_parametrization(
        (-1, 2, 0, -2),
        (-3, -3, -2, -3),
        (-0.0001, -1000, -10000, 1000),
        (-1e-9, 1e9, -1e9, 1e9),
        (-1, -3, 1, 1),
    ),
)
def test_linesearch_finds_boundary_on_flipped_quadratic_function(
    method: type[Linesearch], k: float, c: float, x_0: float, x_1: float
) -> None:
    """Linesearch should easily find minimum of quadratic function."""
    assert k < 0, "Invalid dataset, k need to be positive!"
    f = Function(lambda x: (x - x_0) * (x - x_1) + c, dim_in=1, dim_out=1)
    attempt = (
        method()
        .set_function(f)
        .set_interval(float_to_vec(x_0), float_to_vec(x_1))
        .solve()
    )
    assert attempt - x_0 < 1e-9 or attempt - x_1 < 1e-9
