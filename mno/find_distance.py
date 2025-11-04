from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

import numpy as np

from mno.function import Function
from mno.my_types import Vec, float_to_vec


class FindDistance(ABC):
    """Finds optimal interval on which to find minimum."""

    def with_direction(
        self, function: Function, point: Vec, direction: Vec
    ) -> tuple[Vec, Vec]:
        """Find optimal interval given by two points A B to find minimum on."""

        def help(a):
            print(a, direction, point)
            print(a * direction)
            out = function(a * direction + point)
            print(out)
            return out

        helper = Function(help, dim_in=1, dim_out=1)
        out = self._call(helper)
        print(len(direction), len(point))
        return (point + out[0] * direction, point + out[1] * direction)

    def __call__(self, function: Function) -> tuple[Vec, Vec]:
        """
        Find optimal interval [0,a] to find minimum on.

        Function needs to be R->R.

        Returns a.
        """
        assert function.get_dim() == (1, 1), (
            "Find distance should be called with R -> R functions only!"
        )
        return self._call(function)

    @abstractmethod
    def _call(self, function: Function) -> tuple[Vec, Vec]: ...


def step_doubling(condition: Callable[[Vec], bool], value: Vec) -> Vec:
    """Find "minimal" solution to condition via step doubling."""
    while not condition(value):
        assert not np.isinf(value), "Infinity!"
        value *= 2
    return value


def step_halving(condition: Callable[[Vec], bool], value: Vec) -> Vec:
    """Find "maximal" solution to condition via step halving."""
    while not condition(value):
        if value < 1e-15:
            return float_to_vec(0)
        value /= 2
    return value


class AmijoTest(FindDistance):
    def __init__(self, epsilon: float = 0.0001):
        """Epsilon is a magic constant."""
        assert 0 < epsilon < 1
        self.epsilon = epsilon

    @override
    def _call(self, function: Function) -> tuple[Vec, Vec]:
        alpha = float_to_vec(1)
        basepoint = function(float_to_vec(0))
        slope = self.epsilon * function.grad()(float_to_vec(0))
        if slope > 0:
            alpha *= -1

        def condition_a(alpha: Vec) -> bool:
            return all(function(alpha) < basepoint + slope * alpha)

        def condition_b(alpha: Vec) -> bool:
            return condition_a(alpha) and not (condition_a(alpha * 2))

        if condition_a(alpha):
            alpha = step_doubling(condition_b, alpha)
        else:
            alpha = step_halving(condition_b, alpha)

        return float_to_vec(0), alpha


class GoldsteinTest(FindDistance):
    def __init__(self, epsilon: float = 0.0001):
        """Epsilon is a magic constant."""
        assert 0 < epsilon < 0.5
        self.epsilon = epsilon

    def _call(self, function: Function) -> tuple[Vec, Vec]:
        basepoint = function(float_to_vec(0))
        slope = function.grad()(float_to_vec(0))

        def condition_a(alpha: Vec) -> bool:
            return all(function(alpha) < basepoint + self.epsilon * slope * alpha)

        def condition_b(alpha: Vec) -> bool:
            return condition_a(alpha) and not (condition_a(alpha * 2))

        def condition_c(alpha: Vec) -> bool:
            return all(function(alpha) > basepoint + (1 - self.epsilon) * slope * alpha)

        def condition_d(alpha: Vec) -> bool:
            return condition_c(alpha) and not (condition_c(alpha / 2))

        alpha = float_to_vec(2)
        beta = float_to_vec(1)
        if slope > 0:
            alpha *= -1
            beta *= -1

        if condition_a(alpha):
            alpha = step_doubling(condition_b, alpha)
        else:
            alpha = step_halving(condition_b, alpha)
        if condition_c(beta):
            beta = step_halving(condition_d, beta)
        else:
            beta = step_doubling(condition_d, beta)

        return beta, alpha
