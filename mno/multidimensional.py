from __future__ import annotations

from typing import cast
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from mno.function import Function
from mno.linesearch import GoldenSearch, Linesearch
from mno.my_types import Vec, norm
from mno.stopping_condition import GradientNormCondition, StoppingCondition
from mno.find_distance import FindDistance

class MultidimensionalOptimalization(ABC):
    """Baseclass for multidimensional optimalizations."""

    def __init__(self):
        """Create multidimensional optimalization method."""
        self._function = None
        self._point = None
        self._logged_points = []
        self._stopping_condition = GradientNormCondition()
        self._linesearch = GoldenSearch()

    def set_function(self, function: Function) -> MultidimensionalOptimalization:
        """Set function to optimize."""
        if function.get_dim()[1] != 1:
            warnings.warn(
                "Provided method's image is not R, so we will optimize it's L-2 norm",
                stacklevel=2,
            )
            function = Function(
                lambda a: norm(function(a)), dim_in=function.get_dim()[0], dim_out=1
            )
        self._function = function

        return self

    def set_point(self, point: Vec) -> MultidimensionalOptimalization:
        """Set starting point."""
        assert self._function, "You need to first set function."
        assert len(point) == self._function.get_dim()[0]
        self._point = point
        return self

    def _log_point(self, *points: Vec) -> None:
        self._logged_points.append(points)

    def draw(self) -> None:
        """Draw metrics for the solver."""
        assert self._function is not None

        logged = np.array(self._logged_points)
        _, ax = plt.subplots(2)
        gradiant = self._function.grad()
        ax[0].set_title("Gradient")
        ax[1].set_title("")

        x = list(range(len(logged)))
        grad: list[np.floating] = []
        values: list[np.floating] = []

        for i in range(logged.shape[1]):
            for point in logged[:, i]:
                grad.append(np.linalg.norm(gradiant(point)))
                values.append(self._function(point)[0])
            ax[0].plot(x, grad)
            ax[1].plot(x, values)
        plt.show()

    def set_distance_finder(self, distance_finder: FindDistance) -> MultidimensionalOptimalization:
        self._find_distance  = distance_finder
        return self

    def set_linesearch(self, linesearch: Linesearch) -> MultidimensionalOptimalization:
        """Set linesearch."""
        self.linesearch = linesearch
        return self

    def stopping_condition(
        self, stopping_condition: StoppingCondition
    ) -> MultidimensionalOptimalization:
        """Set stopping condition."""
        self._stopping_condition = stopping_condition
        return self

    def solve(self) -> Vec:
        """Solve the optimalization problem and return the minimal point."""
        assert self._function is not None
        assert self._point is not None
        return self._solve()

    @abstractmethod
    def _solve(self) -> Vec: ...


class CongurateGradient(MultidimensionalOptimalization):
    def _get_direction(self, point: Vec, func: Function) -> Vec: ...
    def _get_step_length(self, point: Vec, direction: Vec, func: Function) -> float:
        self._find_distance(function=)
        self.linesearch.set_interval(point + A0 * direction, point + A1 * direction)

    def _solve(self) -> Vec:
        point: Vec = cast(Vec, self._point)
        i = 0
        func = cast(Function, self._function)
        self.linesearch.set_function(func)
        prev_point = None
        while self._stopping_condition(
            cur_point=point, prev_point=prev_point, function=func, iteration=i
        ):
            prev_point = point
            direction = self._get_direction(point, func)
            point = point + direction * self._get_step_length(point, direction, func)
        return point

class FletcherReeves(CongurateGradient):
    def _get_direction(self, point: Vec, func: Function) -> Vec:

