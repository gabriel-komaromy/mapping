import pylab as pl
from matplotlib import collections as mc


def plot_real_map(walls_list_file):
    with open(walls_list_file, 'r') as f:
        walls_list = f.readlines()
    split_coordinates = [wall.split(',') for wall in walls_list]
    wall_coordinates = []
    for wall in split_coordinates:
        wall_coordinates.append([float(coord.strip()) + 1 for coord in wall])
    wall_coordinates.append([0.9, 0.9, 0.9, 11.1])
    wall_coordinates.append([0.9, 11.1, 11.1, 11.1])
    wall_coordinates.append([11.1, 11.1, 11.1, 0.9])
    wall_coordinates.append([11.1, 0.9, 0.9, 0.9])
    walls = []
    for coord in wall_coordinates:
        walls.append([(coord[0], coord[1]), (coord[2], coord[3])])

    lc = mc.LineCollection(walls, linewidths=2)
    fig, ax = pl.subplots()
    ax.add_collection(lc)
    ax.margins(0.1)
    fig.show()
