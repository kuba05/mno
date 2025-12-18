from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from mno.find_distance import FindDistance, GoldsteinTest
from mno.function import Function
from mno.linesearch import GoldenSearch, Linesearch
from mno.my_types import Vec, array_to_vec, norm
from mno.stopping_condition import (
    GradientNormCondition,
    StoppingCondition,
)


class MultidimensionalOptimalization(ABC):
    """Baseclass for multidimensional optimalizations."""

    def __init__(self):
        """Create multidimensional optimalization method."""
        self._function = None
        self._point = None
        self._logged_points = []
        self._stopping_condition: StoppingCondition = GradientNormCondition()
        self._linesearch: Linesearch = GoldenSearch()
        self._find_distance: FindDistance = GoldsteinTest()  # AmijoTest()

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
        # print("logged points", self._logged_points)

    def draw(self) -> None:
        """Draw metrics for the solver."""
        assert self._function is not None

        logged = np.array(self._logged_points)
        if self._function.get_dim()[0] == 2:
            fig, ax = plt.subplots(3)
            self._draw_for_two_dim(ax[2])
        else:
            fig, ax = plt.subplots(2)
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

    def _draw_for_two_dim(self, axis) -> None:
        assert self._function is not None
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        x, y = np.meshgrid(x, y)
        print(x)
        z = np.array(
            [
                [
                    self._function(array_to_vec([x[i, j], y[i, j]]))[0]
                    for j in range(x.shape[1])
                ]
                for i in range(x.shape[0])
            ]
        )
        z_log = np.log2(z.min() + z + 1)
        print(z.shape)
        contour = axis.contourf(x, y, z_log, levels=100, cmap="viridis")
        plt.colorbar(contour, label="Function Value logged (Z)")
        axis.set_xlabel("X axis")
        axis.set_ylabel("Y axis")
        x = []
        y = []
        for points in self._logged_points:
            axis.scatter(*zip(*points))
            dx, dy = zip(*points)
            x += dx
            y += dy
        axis.plot(x, y, color="gray")

    def set_distance_finder(
        self, distance_finder: FindDistance
    ) -> MultidimensionalOptimalization:
        """Set distance finder for linesearch bounds."""
        self._find_distance = distance_finder
        return self

    def set_linesearch(self, linesearch: Linesearch) -> MultidimensionalOptimalization:
        """Set linesearch."""
        self._linesearch = linesearch
        return self

    def set_stopping_condition(
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
    def _getstep(self, point: Vec, func: Function) -> Vec: ...

    def _solve(self) -> Vec:
        point: Vec = cast(Vec, self._point)
        i = 0
        func = cast(Function, self._function)
        self._linesearch.set_function(func)
        prev_point = None
        while not self._stopping_condition(
            cur_point=point, prev_point=prev_point, function=func, iteration=i
        ):
            self._log_point(point)
            prev_point = point
            point = self._getstep(point, func)
            i += 1
            print(i)
        self._log_point(point)
        return point


def descend_step(
    point: Vec, func: Function, distance_finder: FindDistance, linesearch: Linesearch
) -> Vec:
    """Do one step of congurated gradients method."""
    direction = -func.grad()(point)
    d = distance_finder.with_direction(func, point, direction)
    return linesearch.set_interval(*d).solve()


def conjugateStep(
    point: Vec,
    func: Function,
    gradients: list[Vec],
    prev_direction: Vec,
    distance_finder: FindDistance,
    linesearch: Linesearch,
) -> Vec:
    """Conjugate gradient step."""
    down = (gradients[-1] - gradients[-2]).dot(prev_direction)
    if abs(down) < 1e-9:
        print("Gradients are low", gradients, prev_direction)
    beta = (gradients[-1] - gradients[-2]).dot(gradients[-1]) / down
    direction = -gradients[-1] - beta * prev_direction
    return linesearch.set_interval(
        *distance_finder.with_direction(func, point, direction)
    ).solve()


def DFP_step(
    point: Vec,
    func: Function,
    matrix_s: np.ndarray,
    distance_finder: FindDistance,
    linesearch: Linesearch,
):
    g = func.grad()(point)
    direction = -matrix_s.dot(g)
    d = distance_finder.with_direction(func, point, direction)
    new_point = linesearch.set_interval(*d).solve()
    new_g = func.grad()(new_point)
    p = new_point - point
    q = new_g - g
    p = p.reshape((-1, 1))
    q = q.reshape((-1, 1))

    new_matrix = (
        matrix_s
        + p @ p.T / (q.T @ p)
        - (matrix_s @ q @ q.T @ matrix_s.T) / (q.T @ matrix_s @ q)
    )
    print(new_matrix)
    return new_point, new_matrix


def BFGS_step(
    point: Vec,
    func: Function,
    matrix_s: np.ndarray,
    distance_finder: FindDistance,
    linesearch: Linesearch,
):
    g = func.grad()(point)
    direction = -matrix_s.dot(g)
    d = distance_finder.with_direction(func, point, direction)
    new_point = linesearch.set_interval(*d).solve()
    new_g = func.grad()(new_point)
    s = new_point - point
    y = new_g - g
    s = s.reshape((-1, 1))
    y = y.reshape((-1, 1))
    print(s, y, s.T @ y)

    print((s.T @ y + y.T @ matrix_s @ y), "part1")
    print((s @ s.T), "part2")

    new_matrix = (
        matrix_s
        + (s.T @ y + y.T @ matrix_s @ y) * (s @ s.T) / (s.T @ y) ** 2
        - (matrix_s @ y @ s.T + s @ y.T @ matrix_s) / (s.T @ y)
    )
    print(new_matrix)
    return new_point, new_matrix


class GradientDescend(MultidimensionalOptimalization):
    """Basic gradient descend method."""

    def _getstep(self, point: Vec, func: Function) -> Vec:
        return descend_step(point, func, self._find_distance, self._linesearch)


class ConjugateGradient(MultidimensionalOptimalization):
    """Conjugate gradient method Hestenes Stiefel style."""

    def __init__(self):
        super().__init__()
        self.prevG: list[Vec] = []
        self.prev_direction: Vec | None = None

    def _getstep(self, point: Vec, func: Function) -> Vec:
        # restart every 2*N steps
        if len(self.prevG) >= len(point) * 2:
            self.prevG = []

        self.prevG.append(func.grad()(point))
        if len(self.prevG) > 1 and all(self.prevG[-1] == self.prevG[-2]):
            warnings.warn(
                "Last step was zero. Probably didn't choose a direction of descend."
                "Attempting to fix by using gradient descend."
            )
            self.prevG = self.prevG[-1:]

        # not enough values to make a conjugated step
        if len(self.prevG) == 1 or self.prev_direction is None:
            new_point = descend_step(point, func, self._find_distance, self._linesearch)
            print("descend step")
        else:
            new_point = conjugateStep(
                point,
                func,
                self.prevG,
                self.prev_direction,
                self._find_distance,
                self._linesearch,
            )
            print("conjugate step")
        self.prev_direction = new_point - point
        return new_point


class BFGSMethod(MultidimensionalOptimalization):
    """BFGS method."""

    def __init__(self):
        super().__init__()
        self.matrix: np.ndarray | None = None

    def _getstep(self, point: Vec, func: Function) -> Vec:
        if self.matrix is None:
            self.matrix = np.identity(func.get_dim()[0])
        new_point, new_matrix = BFGS_step(
            point, func, self.matrix, self._find_distance, self._linesearch
        )
        self.matrix = new_matrix
        return new_point


class DFPMethod(MultidimensionalOptimalization):
    """BFGS method."""

    def __init__(self):
        super().__init__()
        self.matrix: np.ndarray | None = None

    def _getstep(self, point: Vec, func: Function) -> Vec:
        if self.matrix is None:
            self.matrix = np.identity(func.get_dim()[0])
        new_point, new_matrix = DFP_step(
            point, func, self.matrix, self._find_distance, self._linesearch
        )
        self.matrix = new_matrix
        return new_point


class BroydenMethod(MultidimensionalOptimalization):
    """BroydenMethod."""

    def __init__(self, value=0.5):
        super().__init__()
        self.value = value
        self.bfgs: np.ndarray | None = None
        self.dfp: np.ndarray | None = None

    def _getstep(self, point: Vec, func: Function) -> Vec:
        if self.bfgs is None:
            self.bfgs = np.identity(func.get_dim()[0])
        if self.dfp is None:
            self.dfp = np.identity(func.get_dim()[0])
        new_point, new_dfp = DFP_step(
            point, func, self.dfp, self._find_distance, self._linesearch
        )
        new_point, new_bfgs = BFGS_step(
            point, func, self.bfgs, self._find_distance, self._linesearch
        )
        self.dfp = new_dfp
        self.bfgs = new_bfgs
        return new_point
