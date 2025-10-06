import numpy as np
import numpy.typing as npt

TYPE = float
Vec = npt.NDArray


def float_to_vec(x: TYPE) -> Vec:
    """Convert float to Vec."""
    return np.array([x])


def array_to_vec(x: list | tuple) -> Vec:
    """Convert array to Vec."""
    return np.array(x)


def dot_prorduct(x: Vec, y: Vec) -> Vec:
    """Return the dot product of two vectors."""
    return np.array(x.dot(y))
