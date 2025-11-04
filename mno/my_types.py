import numpy as np
import numpy.typing as npt

TYPE = float
Vec = npt.NDArray


def float_to_vec(x: TYPE | np.floating) -> Vec:
    """Convert float to Vec."""
    return np.array([x], dtype=TYPE)


def array_to_vec(x: list | tuple) -> Vec:
    """Convert array to Vec."""
    return np.array(x, dtype=TYPE)


def dot_prorduct(x: Vec, y: Vec) -> Vec:
    """Return the dot product of two vectors."""
    return np.array(x.dot(y))


def norm(x: Vec) -> Vec:
    """Return the square of the L2 norm."""
    return dot_prorduct(x, x)
