import numpy as np

from mno.find_distance import StrongWolfe
from mno.function import Function
from mno.linesearch import GoldenSearch, TernarySearch
from mno.my_types import array_to_vec
from mno.stopping_condition import GradientNormCondition, SmallChangeCondition
from mno.multidimensional import ConjugateGradient, GradientDescend


def main() -> None:
    mainTwoD()


def mainOneD() -> None:
    """Run the whole thing."""
    x = (
        GoldenSearch()
        .set_function(Function(lambda a: a[0] ** 2, dim_in=1, dim_out=1))
        .set_stopping_condition(stopping_condition=SmallChangeCondition())
        .set_interval(
            point_a=array_to_vec([-4]),
            point_b=array_to_vec([4]),
        )
    )
    print(x.solve())
    x.draw_picture()

    x = (
        GoldenSearch()
        .set_function(
            Function(
                lambda a: (a[0] - 2) ** 2 + 0.01 * (a[0] - 2) * a[0],
                dim_in=1,
                dim_out=1,
            )
        )
        .set_stopping_condition(stopping_condition=SmallChangeCondition())
        .set_interval(
            point_a=array_to_vec([-4]),
            point_b=array_to_vec([4]),
        )
    )
    print(x.solve())
    x.draw_picture()

    x = (
        GoldenSearch()
        .set_function(
            Function(
                lambda a: -(a[0] - np.sin(a[0])) * np.e ** (a[0] ** 2),
                dim_in=1,
                dim_out=1,
            )
        )
        .set_stopping_condition(stopping_condition=SmallChangeCondition())
        .set_interval(
            point_a=array_to_vec([-4]),
            point_b=array_to_vec([4]),
        )
    )
    print(x.solve())
    x.draw_picture()

    x = (
        GoldenSearch()
        .set_function(
            Function(
                lambda a: -a[0] * np.sin(a[0] ** 2),
                dim_in=1,
                dim_out=1,
            )
        )
        .set_stopping_condition(stopping_condition=SmallChangeCondition())
        .set_interval(
            point_a=array_to_vec([-4]),
            point_b=array_to_vec([4]),
        )
    )
    print(x.solve())
    x.draw_picture()
    """x = (
        GoldenSearch()
        .set_function(
            Function(lambda a: ((a[0] + 2) ** 4) - a[1] ** 2, dim_in=2, dim_out=1)
        )
        .set_stopping_condition(stopping_condition=SmallChangeCondition())
        .set_interval(
            point_a=array_to_vec([0, 0]),
            point_b=array_to_vec([-3, 0]),
        )
    )
    print(x.solve())
    x.draw_picture()"""


def mainTwoD() -> None:
    # method = GradientDescend()
    method = ConjugateGradient()
    function = Function(dim_in=2, dim_out=1, function=lambda a: a[0] ** 2 + a[1] ** 2)
    print(
        method.set_function(function)
        # .set_stopping_condition(GradientNormCondition(1e-22))
        .set_stopping_condition(SmallChangeCondition())
        # .set_distance_finder(StrongWolfe())
        .set_point(array_to_vec([-3, 1]))
        .solve()
    )
    method.draw()
    function = Function(
        dim_in=2,
        dim_out=1,
        function=lambda a: (1.5 - a[0] + a[0] * a[1]) ** 2
        + (2.25 - a[0] + a[0] * a[1] ** 2) ** 2
        + (2.625 - a[0] + a[0] * a[1] ** 3) ** 2,
        derivative=lambda a: array_to_vec(
            [
                2 * (1.5 - a[0] + a[0] * a[1]) * (a[1] - 1)
                + 2 * (2.25 - a[0] + a[0] * a[1] ** 2) * (a[1] ** 2 - 1)
                + 2 * (2.625 - a[0] + a[0] * a[1] ** 3) * (a[1] ** 3 - 1),
                (1.5 - a[0] + a[0] * a[1]) * 2 * a[0]
                + (2.25 - a[0] + a[0] * a[1] ** 2) * 4 * a[0] * a[1]
                + (2.625 - a[0] + a[0] * a[1] ** 3) * 6 * a[0] * a[1] ** 2,
            ]
        ),
    )
    print(
        method.set_function(function)
        .set_stopping_condition(GradientNormCondition(1e-9))
        # .set_stopping_condition(SmallChangeCondition())
        # .set_distance_finder(StrongWolfe())
        .set_point(array_to_vec([-3, 1]))
        .solve()
    )
    method.draw()


if __name__ == "__main__":
    main()
