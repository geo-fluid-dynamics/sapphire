import numpy as np


def magnitude(vector):

    return np.sqrt(np.sum(np.square(vector)))


def normalize_to_unit_vector(vector):

    return tuple(vector/magnitude(vector))
