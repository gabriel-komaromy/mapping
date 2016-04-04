import numpy as np
from random import Random


def forget_map_context(proba_map, observed_map):
    # has to be revised if not square maps
    dimension = len(proba_map)
    for x in xrange(dimension):
        for y in xrange(dimension):
            if observed_map[x, y] == 0:
                proba_map[x, y] = np.NaN

    low_row = find_lowest_row(proba_map, dimension)
    high_row = find_highest_row(proba_map, dimension)
    low_col = find_lowest_col(proba_map, dimension)
    high_col = find_highest_col(proba_map, dimension)

    map_slice = proba_map[low_row:high_row, low_col:high_col]
    rand = Random()
    rotations = rand.randint(0, 3)
    rotated = np.rot90(map_slice, rotations)

    return rotated


def find_lowest_row(proba_map, dimension):
    """Can you tell I wrote this at 1:30am?"""
    for x in xrange(dimension):
        for y in xrange(dimension):
            if not np.isnan(proba_map[x, y]):
                return x


def find_lowest_col(proba_map, dimension):
    for y in xrange(dimension):
        for x in xrange(dimension):
            if not np.isnan(proba_map[x, y]):
                return y


def find_highest_row(proba_map, dimension):
    for x in reversed(xrange(dimension)):
        for y in xrange(dimension):
            if not np.isnan(proba_map[x, y]):
                return x


def find_highest_col(proba_map, dimension):
    for y in reversed(xrange(dimension)):
        for x in xrange(dimension):
            if not np.isnan(proba_map[x, y]):
                return y
