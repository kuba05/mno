import pytest

from mno.function import Function
from mno.my_types import Vec, float_to_vec, array_to_vec
from mno.find_distance import FindDistance, AmijoTest, GoldsteinTest

methods = [AmijoTest(), GoldsteinTest()]


def add_methods_into_parametrization(*parametrization: tuple) -> list[tuple]:
    """Replicates parametrization for each method."""
    return [(method, *x) for x in parametrization for method in methods]


@pytest.mark.parametrize(
    ("method", "function"),
    add_methods_into_parametrization(
        (lambda a: a**4 - a * 3,),
        (lambda a: (a - 20) ** 4 - (a - 20) * 3,),
        (lambda a: 3**a,),
    ),
)
def test_start_is_under_origin(method: FindDistance, function: callable) -> None:
    """Midpoint of the found interval should always be lower than the origin."""
    f = Function(function, dim_in=1, dim_out=1)
    a, b = method(f)
    assert f(a) <= f(float_to_vec(0))


@pytest.mark.parametrize(
    ("method", "function"),
    add_methods_into_parametrization(
        (lambda a: a**4 - a * 3,),
        (lambda a: (a - 20) ** 4 - (a - 20) * 3,),
        (lambda a: 3**a,),
    ),
)
def test_endpoint_is_under_origin(method: FindDistance, function: callable) -> None:
    """Midpoint of the found interval should always be lower than the origin."""
    f = Function(function, dim_in=1, dim_out=1)
    a, b = method(f)
    assert f(b) <= f(float_to_vec(0))


@pytest.mark.parametrize(
    ("method", "function", "point", "direction"),
    add_methods_into_parametrization(
        (lambda a: a[2] ** 4 - a[1] * 3, [3, 41, 2], [2, 5, 7]),
        (lambda a: (a[1] - 20) ** 4 - (a[0] - 20) * 3, [12, 42], [-4, -5]),
        (lambda a: 3 ** a[0], [0], [1]),
    ),
)
def test_endpoint_is_under_origin_for_multidim(
    method: FindDistance, function: callable, point: list, direction: list
) -> None:
    """Midpoint of the found interval should always be lower than the origin."""

    f = Function(function, dim_in=len(point), dim_out=1)
    a, b = method.with_direction(f, array_to_vec(point), array_to_vec(direction))
    assert f(b) <= f(array_to_vec(point))


@pytest.mark.parametrize(
    ("method", "function", "point", "direction"),
    add_methods_into_parametrization(
        (lambda a: a[2] ** 4 - a[1] * 3, [3, 41, 2], [2, 5, 7]),
        (lambda a: (a[1] - 20) ** 4 - (a[0] - 20) * 3, [12, 42], [-4, -5]),
        (lambda a: 3 ** a[0], [0], [1]),
    ),
)
def test_start_is_under_origin_for_multidim(
    method: FindDistance, function: callable, point: list, direction: list
) -> None:
    """Midpoint of the found interval should always be lower than the origin."""
    f = Function(function, dim_in=len(point), dim_out=1)
    a, b = method.with_direction(f, array_to_vec(point), array_to_vec(direction))
    assert f((a + b) / 2) <= f(array_to_vec(point))
