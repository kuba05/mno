from __future__ import annotations

from collections.abc import Callable

import numpy as np

from mno.my_types import TYPE, Vec


class Function:
    """
    Function R^N -> R^M.

    Can also have derivative (defaults to None -> derivative is computed numerically).
    """

    def __init__(
        self,
        function: Callable[[Vec], Vec],
        dim_in: int,
        dim_out: int,
        derivative: Callable[[Vec], Vec] | None = None,
    ):
        """
        Create a general function over R.

        Function is a the callable function itself. It should take a single argument
        a vector of dimension dim_in and output a vector of dimension dim_out.

        An optional argument, derivative, can also be provided. It should be a function:
        R^dim_in -> R^(dim_in*dim_out)

        If it is not provided, numerical aproximation is used when necessary.
        """
        self._function = function
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._derivative = derivative

    def __call__(self, point: Vec) -> Vec:
        """Return function's value at a given point."""
        assert len(point) == self._dim_in, (
            f"When calling function, dimension of the argument was off. Expected {self._dim_in}, got {len(point)}."
        )
        return self._function(point)

    def has_derivative(self) -> bool:
        """Check if function has symbolic derivative."""
        return self._derivative is not None

    def grad(self) -> Function:
        """
        Find function's gradient, either symbolic (if possible) or numerical.

        Note gradient is always written out as a vector, where components are written
        after each other.
        """
        if self._derivative is not None:
            return Function(
                self._derivative,
                dim_in=self._dim_in,
                dim_out=self._dim_out * self._dim_in,
                derivative=None,
            )
        return self.numerical_grad()

    def get_dim(self) -> tuple[int, int]:
        """Return the in dimension and out dimension of function."""
        return (self._dim_in, self._dim_out)

    def numerical_grad(self, step: float = 0.01) -> Function:
        """
        Find function's numerical gradient.

        Uses central derivative with given step.
        """

        def helper(point: Vec) -> Vec:
            return (
                (
                    np.array(
                        [
                            self._function(
                                point + step * np.eye(len(point), dtype=TYPE)
                            )
                            for _ in range(len(point))
                        ]
                    )
                    - np.array(
                        [
                            self._function(
                                point - step * np.eye(len(point), dtype=TYPE)
                            )
                            for _ in range(len(point))
                        ]
                    )
                ).reshape(-1)
                / 2
                / step
            )

        return Function(
            helper,
            dim_in=self._dim_in,
            dim_out=self._dim_out * self._dim_in,
            derivative=None,
        )
