from typing import override

import numpy as np
from function import Function
from my_types import Vec


class StoppingCondition:
    """Abstract for a stopping condition for a numerical method."""

    def __init__(self, critical_value: float = 1e-9):
        """Stoping condition with some critical value."""
        self._critical_value = critical_value

    def _should_stop(
        self,
        _function: Function,
        _cur_point: Vec,
        _prev_point: Vec | None,
        iteration: int,
    ) -> bool:
        raise NotImplementedError

    def __call__(
        self, function: Function, cur_point: Vec, prev_point: Vec | None, iteration: int
    ) -> bool:
        """
        Check if the method should stop.

        Returns true if it should stop.

        Recieves the function, for which minimum is to be found,
        and point at which the function is to be run.
        """
        return self._should_stop(function, cur_point, prev_point, iteration)


class IterationCondition(StoppingCondition):
    """Stopping condition checking gradient norm."""

    @override
    def _should_stop(
        self, function: Function, cur_point: Vec, prev_point: Vec | None, iteration: int
    ) -> bool:
        return iteration >= 10**3


class GradientNormCondition(StoppingCondition):
    """Stopping condition checking gradient norm."""

    @override
    def _should_stop(
        self, function: Function, cur_point: Vec, prev_point: Vec | None, iteration: int
    ) -> bool:
        if iteration >= 10**5:
            return True
        grad = function.grad()(cur_point)
        return np.dot(grad, grad) < self._critical_value


class SmallChangeCondition(StoppingCondition):
    """Stopping condition checking change since last iteration."""

    @override
    def _should_stop(
        self, function: Function, cur_point: Vec, prev_point: Vec | None, iteration: int
    ) -> bool:
        if iteration >= 10**5:
            return True

        if prev_point is None:
            return False

        change = cur_point - prev_point
        return np.dot(change, change) < self._critical_value
