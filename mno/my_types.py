import numpy.typing as npt
import numpy as np

TYPE = float
Vec = npt.NDArray


def float_to_vec(x: float) -> Vec:
    """Convert float to Vec."""
    return np.array([x])


def array_to_vec(x: list | tuple) -> Vec:
    """Convert array to Vec."""
    return np.array(x)
