import pytest

from mno.linesearch import Linesearch
from mno.function import Function

methods = []


@pytest.mark.parametrize(("a", "b", "x_0", "x_1"), (1, 0, -2, 3))
def test_linesearch_improves_solution_on_linear_function(
    method: Linesearch, k: float, sol: float, x_0: float, x_1: float
):
    assert x_0 <= sol <= x_1, (
        "Invalid test data - solution not contained in guess interval"
    )
    assert x_0 <= x_1, "Invalid test data - guess interval of negative size"
    f = Function(lambda x: (x - sol) * k)
    assert method(f, x_0, x_1) - sol < 1e-9
