import numpy as np


def merge_maps(maps):
    print similarity_score(maps[1], maps[2])


def similarity_score(map1, map2):
    f1 = map1.flatten()
    u1 = np.unique(f1[~np.isnan(f1)])
    f2 = map2.flatten()
    u2 = np.unique(f2[~np.isnan(f2)])
    c_values = np.unique(np.concatenate((u1, u2)))
    accum = 0
    for value in c_values:
        accum += distance(map1, map2, value) + distance(map2, map1, value)
    return accum


def distance(map1, map2, value):
    if not np.isnan(value):
        accum = []
        for x1 in xrange(map1.shape[0]):
            for y1 in xrange(map1.shape[1]):
                if map1[x1, y1] == value:
                    matching = []
                    for x2 in xrange(map2.shape[0]):
                        for y2 in xrange(map2.shape[1]):
                            if map2[x2, y2] == value:
                                matching.append(
                                    manhattan_distance((x1, y1), (x2, y2))
                                    )
                    if len(matching) > 0:
                        accum.append(min(matching))
                    else:
                        accum.append(0)
        if len(accum) == 0:
            return 0
        else:
            return sum(accum) / len(accum)
    else:
        return 0


def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
