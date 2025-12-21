import math
from typing import Callable

import numpy as np
import sympy
import matplotlib.pyplot as plt

from mno.find_distance import StrongWolfe
from mno.function import Function
from mno.linesearch import GoldenSearch, TernarySearch
from mno.multidimensional import BroydenMethod, GradientDescend
from mno.my_types import Vec, array_to_vec, float_to_vec
from mno.stopping_condition import GradientNormCondition, SmallChangeCondition


def main() -> None: ...


def arbitrary_function_broyden(
    starting_point: list[float],
    function: Callable[[np.ndarray], float],
    derivative: Callable[[np.ndarray], list[float]] | None = None,
    distance_finder_epsilon: float = 0.0001,
    linesearch_max_iter: int = 1000,
    linesearch_critical: float = 1e-8,
    max_iter: int = 1000,
    critical: float = 0.0001,
    num_step: float = 0.01,
    draw: bool = True,
) -> float | None:
    """
    Find the minimum of an arbitrary function using the broyden's method.

    Function needs to be a callable. If derivative isn't provided, it is computed numerically.
    """
    broyden = (
        BroydenMethod()
        .set_distance_finder(StrongWolfe(epsilon=distance_finder_epsilon))
        .set_linesearch(
            GoldenSearch().set_stopping_condition(
                SmallChangeCondition(
                    max_iterations=linesearch_max_iter,
                    critical_value=linesearch_critical,
                )
            )
        )
        .set_stopping_condition(
            GradientNormCondition(max_iterations=max_iter, critical_value=critical)
        )
    )

    def helper(a: Vec) -> Vec:
        return float_to_vec(function(a))

    def der_helper(a: Vec) -> Vec:
        if derivative is None:
            raise ValueError
        return array_to_vec(derivative(a))

    broyden.set_function(
        Function(
            function=helper,
            derivative=der_helper if derivative is not None else None,
            dim_in=len(starting_point),
            dim_out=1,
            numerical_derivative_step=num_step,
        )
    ).set_point(array_to_vec(starting_point))

    if draw:
        try:
            print("Broyden found the following solution:", broyden.solve()[0])
        except ValueError as e:
            print("Broyden threw the following error:", e)
        broyden.draw()
        plt.show()
        return None
    else:
        try:
            broyden.solve()
        except ValueError as e:
            ...
        return broyden.get_iters_needed()


def arbitrary_function_hestenes(
    starting_point: list[float],
    function: Callable[[np.ndarray], float],
    derivative: Callable[[np.ndarray], list[float]] | None = None,
    distance_finder_epsilon: float = 0.0001,
    linesearch_max_iter: int = 1000,
    linesearch_critical: float = 1e-8,
    max_iter: int = 1000,
    critical: float = 0.0001,
    num_step: float = 0.01,
    draw: bool = True,
) -> float | None:
    """
    Find the minimum of an arbitrary function using the hestene's methods.

    Function needs to be a callable. If derivative isn't provided, it is computed numerically.
    """
    hestenes = (
        GradientDescend()
        .set_distance_finder(StrongWolfe(epsilon=distance_finder_epsilon))
        .set_linesearch(
            GoldenSearch().set_stopping_condition(
                SmallChangeCondition(
                    max_iterations=linesearch_max_iter,
                    critical_value=linesearch_critical,
                )
            )
        )
        .set_stopping_condition(
            GradientNormCondition(max_iterations=max_iter, critical_value=critical)
        )
    )

    def helper(a: Vec) -> Vec:
        return float_to_vec(function(a))

    def der_helper(a: Vec) -> Vec:
        if derivative is None:
            raise ValueError
        return array_to_vec(derivative(a))

    hestenes.set_function(
        Function(
            function=helper,
            derivative=der_helper if derivative is not None else None,
            dim_in=len(starting_point),
            dim_out=1,
            numerical_derivative_step=num_step,
        )
    ).set_point(array_to_vec(starting_point))
    if draw:
        try:
            print("Hestenes found the following solution:", hestenes.solve())
        except ValueError as e:
            print("Hestenes threw the following error:", e)
        hestenes.draw()
        plt.show()
        return None
    else:
        try:
            hestenes.solve()
        except ValueError as e:
            ...
        return hestenes.get_iters_needed()


def autocompute_derivative(function, dims):
    """Automatically computes exact derivative of given function."""
    variables = sympy.symbols(f"a0:{dims}")
    derivs = [function(variables).diff(var) for var in variables]

    def helper(a: Vec) -> list[float]:
        return [deriv.subs(list(zip(variables, a, strict=True))) for deriv in derivs]

    return helper


def f2(u: Vec) -> float:
    x = u[0]
    y = u[1]
    return (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )


def p3(u: Vec) -> float:
    """The chosen 3d function - perm function with d =3, beta = 1."""
    suma = 0
    for i in range(3):
        s = 0
        for j in range(3):
            s += (j + 2) * (u[j] ** (1 + i) - 1 / ((j + 1) ** (1 + i)))
        suma += s**2
    return suma


def H4(u: Vec) -> float:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    a = np.array(
        [
            [10.0, 3.0, 17.0, 3.5],
            [0.05, 10.0, 17.0, 0.1],
            [3.0, 3.5, 1.7, 10.0],
            [17.0, 8.0, 0.05, 10.0],
        ]
    )

    p = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124],
            [0.2329, 0.4135, 0.8307, 0.3736],
            [0.2348, 0.1451, 0.3522, 0.2883],
            [0.4047, 0.8828, 0.8732, 0.5743],
        ]
    )
    outer_sum = 0.0
    for i in range(4):
        inner_sum = 0.0
        for j in range(4):
            inner_sum += a[i, j] * (u[j] - p[i, j]) ** 2
        outer_sum += alpha[i] * np.e ** (-inner_sum)
    return 1 / 0.839 * (1.1 - outer_sum)


def main_test_h4() -> None:
    """Run the optimization on Hartmann 4D function."""
    # the given 4D function

    arbitrary_function_broyden(
        [0.25, 0.25, 0.25, 0.25],
        H4,
        linesearch_max_iter=1000,
        max_iter=100,
        derivative=None,  # autocompute_derivative(H4, 4),
        num_step=0.01,
        distance_finder_epsilon=1e-4,
        linesearch_critical=1e-9,
        critical=1e-9,
    )


def main_test_f2() -> None:
    """Run the optimization on the given 2d function."""
    # the given 4D function

    arbitrary_function_hestenes(
        [0.25, 0.25],
        f2,
        linesearch_max_iter=1000,
        max_iter=100,
        derivative=autocompute_derivative(f2, 2),
        num_step=0.01,
        distance_finder_epsilon=1e-4,
        linesearch_critical=1e-9,
        critical=1e-9,
    )


def autotests():
    """
    Just a helper function for me to fillup the tables quickly.

    We can enter a special format of the experiments we want to run and this will run them quickly for us.
    We do need to "manually" set the starting point, method and function in code tho.
    """
    point = [0, 0, 0]
    fce = p3
    print(
        """\\begin{table}[!h]
\\begin{center}
\\caption{Perm function 0,3,1, Hestenes method, point $"""
        + str(point)
        + """$}
\\begin{tabular}{ |c|c|c|c|c|c| } 
\\hline Derivative & FD epsilon & LS crit. val. & Crit. val. & Iterations needed \\\\
\\hline"""
    )
    with open(0) as file:
        for line in file:
            if line.strip() == "":
                print("\\hline ")
                continue
            der, eps, ls_crit, crit = map(float, line.strip().split())
            iters = arbitrary_function_hestenes(
                point,
                fce,
                linesearch_max_iter=1000,
                max_iter=100,
                derivative=None if der != 0 else autocompute_derivative(fce, 3),
                num_step=der,
                distance_finder_epsilon=10**eps,
                linesearch_critical=10**ls_crit,
                critical=10**crit,
                draw=False,
            )
            if iters == None:
                iters = "ERROR"
            if iters == 100:
                iters = "MAX ITERS ($1000$)"
            print(
                f"{der if der != 0 else 'exact'} & $10^{{ {eps} }}$ & $10^{{ {ls_crit} }}$ & $10^{{ {crit} }}$ & {iters} \\\\"
            )

    print("""
\\end{tabular}
\\end{center}
\\end{table}
""")


# Usefull for batch experiments
# autotests()

if __name__ == "__main__":
    # example of how to use
    main_test_f2()
    main_test_h4()
