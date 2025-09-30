import pytest

from mno.function import Function
from mno.linesearch import Linesearch, TernarySearch
from mno.my_types import float_to_vec

methods = [TernarySearch]


par = [(1, 2, 0, -2)]


@pytest.mark.parametrize(
    ("method", "k", "c", "x_0", "x_1"), [(m, *p) for p in par for m in methods]
)
def test_linesearch_improves_solution_on_quadratic_function(
    method: type[Linesearch], k: float, c: float, x_0: float, x_1: float
) -> None:
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
