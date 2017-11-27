""" Defining and plotting the geometry of the plots in E22 """
import os
from plotting_compat import plt
from get_data import parse_filename
from find_plot import find_plot, treatment_names, treatments
import math
import numpy as np
import time

xmin = 599211.37
ymin = 6615133.59
angle = 0.7376


def make_rectangle(x0, y0, angle, W, H):
    """returning x and y, each lists of length 4, coordinates of the
rectangle with width W and height H. Lower left corner at (x0, y0)
before rotation by angle.

    """ 
    s = math.sin(angle)
    c = math.cos(angle)
    x1 = np.array([0, W, W, 0])
    y1 = np.array([0, 0, H, H])
    x = x1 * c - y1 * s + x0
    y = x1 * s + y1 * c + y0
    return list(zip(x.tolist(), y.tolist()))


main_rectangle = make_rectangle(45.5 + xmin, -11.5 + ymin, angle, 86.5, 83)


def divide_rectangle(p, n, other_way=False, gaps=(0, 0, 0)):
    """Divides the rectangle

      p2 --- p1 
      |       |
      p3 --- p0

    (possibly rotated, where pi is (xi, yi),) into n equal
    rectangles. If other_way is false, new points are inserted 
    between p0 and p1, and p3 and p2.
    If gaps are not all zero, there will be gaps like so (for n = 2):

      p2     p1 
                 gap[2]
      r2 --- r1
      |       |
      r3 --- r0 
                 gap[1]
      q3 --- q1 
      |       |
      q2 --- q0 
                 gap[0]
      p3     p0

    """
    def between(p1, p2, i, n):
        f = (i + 1) * 1.0 / n
        x = p1[0] * (1 - f) + p2[0] * f
        y = p1[1] * (1 - f) + p2[1] * f
        return x, y
    r = []
    if other_way:
        p = (p[-1], p[0], p[1], p[2])
    base = [p[0], p[3]]
    for i in range(n):
        new_points = [between(p[0], p[1], i, n), between(p[3], p[2], i, n)]
        r.append([base[0]] + new_points + [base[1]])
        base = new_points
    return r


def plot_rectangle(p, color='k', text=None):
    x, y = list(zip(*p))
    plt.plot(list(x) + [x[0]], list(y) + [y[0]], color)
    if text:
        x, y = rectangle_midpoint(p)
        plt.text(x, y, text, fontsize=8)


def plot_rectangles(rectangles, names=[]):
    if isinstance(rectangles, dict):
        rectangles = list(rectangles.values())
    for i, r in enumerate(rectangles):
        plot_rectangle(r, text=None if not names else names[i])
    plt.axis('equal')


def rectangle_midpoint(p):
    return np.mean(np.array(p), axis=0)


def plot_treatment_text(plots, treatments):
    # plots and treatments are both dicts
    for i, r in plots.items():
        x, y = rectangle_midpoint(r)
        plt.text(x, y, repr(i) + ':' + treatments[i][0] + '.',
                 va='bottom', ha='right', rotation=-45)


def plot_everything(xp, yp, plots_used, treatments):
    plt.plot(xp, yp, '.')
    plt.hold(True)
    plot_rectangles(list(plots_used.values()))
    plot_treatment_text(plots_used, treatments)


def combine_rectangles(rectangle_list):
    p = sum(rectangle_list, [])
    xmin_p = min(p)
    xmax_p = max(p)
    ymin_p = min(p, key=lambda x: x[1])
    ymax_p = max(p, key=lambda x: x[1])
    return [ymax_p, xmax_p, ymin_p, xmin_p]


def small_rectangles():
    # nw = northwest etc
    # pne is closest to the buildings
    pnw, pne, psw = (np.array(main_rectangle[i]) for i in [3, 2, 0])
    # unit vectors
    une = (pne - pnw) / np.linalg.norm(pne - pnw)
    use = (psw - pnw) / np.linalg.norm(psw - pnw)

    def local_to_p(points):
        return [list(pnw + une * p[0] + use * p[1]) for p in points]
    plot_length = np.linalg.norm(pne - pnw) / 32
    width_limits = ([0, 10.8], [10.8, 21.6],
                    [27, 37.8], [37.8, 48.6],
                    [54, 64.8], [64.8, 75.6])
    length_limits = [[i * plot_length, (i + 1) * plot_length]
                     for i in range(32)]
    p = dict()
    offset = 3.7
    for i in range(6):
        for j in range(32):
            v0, v1 = width_limits[i]
            v0 += offset
            v1 += offset
            w0, w1 = length_limits[j]
            p[(i + 1) * 100 + j + 1] = local_to_p([[w0, v0], [w1, v0],
                                                   [w1, v1], [w0, v1]])
    return p


def bigger_rectangle_from_corner(n):
    s = small_rectangles()
    four = [s[i] for i in [n, n + 1, n + 100, n + 101]]
    return combine_rectangles(four)


def all_field_big_rectangles():
    # old and still used version
    large_rectangles = divide_rectangle(main_rectangle, 3, 1)
    plots = []
    for r in large_rectangles:
        plots += divide_rectangle(r, 16, 1)
    return plots

# some of the plots are not numbered the same way always in the result
# files (due to waypoint list errors). I am numbering them simply columnwise,
# starting in northern corner

# todo use the small rectangles (and functions above) to make the migmin_field rectangles
def migmin_field_rectangles():
    plot_indexes = [0, 1, 4, 5, 8, 11, 13, 15, 18, 19, 22, 23, 25, 26, 28, 30, 32,
                    35, 36, 38, 41, 43, 46, 47]
    plots = all_field_big_rectangles()
    plots_used = [plots[i] for i in plot_indexes]
    return {key + 1: x for key, x in enumerate(plots_used)}


def agropro_rectangles():
    keys = [128, 228, 127, 227, 214, 213, 112, 111, 211, 108, 107,
            332, 331, 330, 329, 429, 424, 323, 423, 322, 321, 316, 315, 415, 305, 401,
            528, 628, 527, 627, 522, 622, 521, 621, 518, 517, 617, 508, 507, 606, 505, 605]
    small = small_rectangles()
    return {key: small[key] for key in keys}


if 0:
    q = os.listdir('../results/new_results')
    q2 = os.listdir('../results/results')
    q3 = os.listdir('../results0')
    filenames = list(
        set([x for x in list(set(q + q2 + q3)) if x.startswith('2')]))
    outliers = ['2015-08-31-12-20-49-x599250_238973-y6615154_73666-z0_0_right_Plot_17_',
                '2015-07-09-11-57-33-x599234_384817-y6615178_32899-z0_0_right_Plot_7_']
    for o in outliers:
        filenames.remove(o)
    filenames.sort()
    w = [parse_filename(name)['vehicle_pos'] for name in filenames]
    maxy = max([a['y'] for a in w])
    w = [x for x in w if x['y'] < maxy - 20]
    xp = [a['x'] for a in w]
    yp = [a['y'] for a in w]
    plots_used = migmin_field_rectangles()
    plot_everything(xp, yp, plots_used, treatments)
    plt.axis('equal')
