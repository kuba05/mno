from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import cast, overload

from mno.function import Function
from mno.my_types import Vec, float_to_vec
from mno.stopping_condition import SmallChangeCondition, StoppingCondition


class Linesearch(ABC):
    """Linesearch abc."""

    @overload
    def __init__(self): ...
    @overload
    def __init__(self, function: Function): ...
    @overload
    def __init__(self, function: Function, points: tuple[Vec, Vec]): ...
    def __init__(
        self, function: Function | None = None, points: tuple[Vec, Vec] | None = None
    ):
        """Create a linesearch object."""
        self._stopping_condition = SmallChangeCondition()
        if points is not None:
            assert function is not None, (
                "You need to provide a function to set interval!"
            )
            assert function.get_dim()[0] == len(points[0]), (
                "First point doesn't match dimensions with function!"
            )
            assert function.get_dim()[0] == len(points[1]), (
                "Second point doesn't match dimensions with function!"
            )
        self._function = function
        self._points = points
        self._logged_points = []

    def set_function(self, function: Function) -> Linesearch:
        """Set function to optimize with out dim 1."""
        assert function.get_dim()[1] == 1, "Function doesn't have outdim 1!"
        self._function = function
        return self

    def set_stopping_condition(
        self, stopping_condition: StoppingCondition
    ) -> Linesearch:
        """Set stopping condition for this linesearch."""
        self.stopping_condition = stopping_condition
        return self

    def set_interval(self, point_a: Vec, point_b: Vec) -> Linesearch:
        """Set interval on which linesearch should be done."""
        assert self._function is not None, "You need to first set a function!"
        assert len(point_a) == self._function.get_dim()[0], (
            "Point a isn't a valid argument for set function as their dimensions don't"
            "match!"
        )
        assert len(point_b) == self._function.get_dim()[0], (
            "Point b isn't a valid argument for set function as their dimensions don't"
            "match!"
        )
        self._points = [point_a, point_b]
        return self

    def _log_points(self, *points: Vec) -> None:
        self._logged_points.append(points)

    def draw_picture(self) -> None:
        """Draws picture to ilustrate which points the search tried."""
        x = np.linspace(0, 1, 400)
        line_function = self._get_line_function()
        y = [line_function(float_to_vec(a))[0] for a in x]
        plt.plot(x, y, label="Linesearch function")
        for i, points in enumerate(self._logged_points):
            x = points
            y = [line_function(a) for a in x]
            print(x, y)
            plt.scatter(x, y, s=100, label=f"Generation {i}")
        plt.legend()
        plt.show()

    def _get_line_function(self) -> Function:
        direction = self._points[1] - self._points[0]

        def helper(k: Vec) -> Vec:
            assert self._function is not None
            point = k[0] * direction + self._points[0]
            return self._function(point)

        return Function(function=helper, dim_in=1, dim_out=1)

    def _revert_line_function(self, point: Vec) -> Vec:
        return point * (self._points[1] - self._points[0]) + self._points[0]

    def solve(self) -> Vec:
        """Return a solution to the set function using set stopping condition."""
        assert self._function is not None, "You need to first set a function!"
        assert self._points is not None, "You need to first set points!"
        return self._revert_line_function(self._solve(self._get_line_function()))

    @abstractmethod
    def _solve(self, function: Function) -> Vec:
        """Optimize function R->R on the range (0,1)."""


class TernarySearch(Linesearch):
    """Find minimum using ternary search."""

    def _solve(self, function: Function) -> Vec:
        left = float_to_vec(0)
        right = float_to_vec(1)
        prev_mid_point = None
        i = 0
        while not self._stopping_condition(
            function=function,
            cur_point=(left + right) / 2,
            prev_point=prev_mid_point,
            iteration=i,
        ):
            i += 1
            prev_mid_point = (left + right) / 2
            left_mid = (2 * left + right) / 3
            right_mid = (left + 2 * right) / 3
            self._log_points(left, left_mid, right_mid, right)
            if function(left_mid) > function(right_mid):
                left = left_mid
            else:
                right = right_mid
        print("Iteration", i)

        return (left + right) / 2


class GoldenSearch(Linesearch):
    """Golden ration search."""

    invphi = (np.sqrt(5) - 1) / 2

    def _solve(self, function: Function) -> Vec:
        a = float_to_vec(0)
        d = float_to_vec(1)
        b = d - (d - a) * self.invphi
        c = a + (d - a) * self.invphi
        prev_mid_point = None
        i = 0
        while not self._stopping_condition(
            function=function,
            cur_point=(a + d) / 2,
            prev_point=prev_mid_point,
            iteration=i,
        ):
            self._log_points(a, b, c, d)
            i += 1
            prev_mid_point = (a + d) / 2
            if function(b) < function(c):
                d = c
                c = b
                b = d - (d - a) * self.invphi
            else:
                a = b
                b = c
                c = a + (d - a) * self.invphi
        print("Iteration", i)

        return (a + d) / 2


class NewtonMethod(Linesearch):
    """Netwton's method for linesearch."""

    def _solve(self, function: Function) -> Vec:
        i = 0
        x = float_to_vec(0.5)
        prev_x = None

        def eq(a: Vec) -> Vec:
            return function.grad()(a) / (function.grad().grad()(a))

        while not self._stopping_condition(
            function=function, cur_point=x, prev_point=prev_x, iteration=i
        ):
            i += 1
            prev_x = x
            x = x - eq(x)
            self._log_points(x)
        return x
