import typing
import numpy as np


def magnitude(vector: typing.Iterable) -> float:

    return np.sqrt(np.sum(np.square(vector)))


def normalize_to_unit_vector(vector: np.array) -> typing.Tuple[float]:

    return tuple(vector/magnitude(vector))
