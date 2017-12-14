"""Utilities to make, plot, move, rotate and divide rectangles and
 other 4-polygons (maybe other polygons also).

Used for defining and plotting the geometry of the plots in E22.

A 4-polygon (for example a rectangle) is represented by the corners as
[[x1, x2, x3, x4], [y1, y2, y3, y4]]

"""
from plotting_compat import plt
import numpy as np


def rotate_polygon(polygon, angle, about=0):
    """about is the point the polygon is rotated about, and can be 1) a
    pair [x,y] of coordinates, 2) an integer (zero based) representing
    which coner to rotate about, or 3) 'center' or 'c'

    example: 

    % rotates 45 degrees about (10, 0) 
    r = rotate_polygon([[5, 10 , 10, 5], [0, 0, 1, 1]], np.pi/4, 1) 
    % rotates 45 degrees about center point
    r = rotate_polygon([[5, 10 , 10, 5], [0, 0, 1, 1]], np.pi/4, 'c') 

    """
    # makes a copy so I don't destructively modify polygon:
    x, y = np.array(polygon)
    if about in ('center', 'c'):
        about = (x.mean(), y.mean())
    elif isinstance(about, int):
        about = (x[about], y[about])
    x -= about[0]
    y -= about[1]
    s = np.sin(angle)
    c = np.cos(angle)
    x1 = x * c - y * s + about[0]
    y1 = x * s + y * c + about[1]
    return np.array([x1, y1])


def move_polygon(polygon, x, y):
    polygon = np.array(polygon)
    polygon[0] += x
    polygon[1] += y
    return polygon


def make_rectangle_with_angle(x0, y0, W, H, angle, about=0):
    """returning 2x4 array([x, y]), (two vectors of length 4), coordinates of the
    rectangle with width W and height H. Lower left corner at (x0, y0)
    before rotation by angle about point 'about' (see help(rotate_polygon))

    """
    x = np.array([0, W, W, 0]) + x0
    y = np.array([0, 0, H, H]) + y0
    return rotate_polygon([x, y], angle, about)


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
    if len(p[0]) != 2:  # todo rydde opp hvor dette kalles fra
        p = list(zip(*p))
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
    if callable(p):
        # if rectangles is not a list of rectangles, rectangle, but a
        # callable function of the dataframe row, just return
        return 1
    x, y = list(zip(*p))
    plt.plot(list(x) + [x[0]], list(y) + [y[0]], color)
    if text:
        x, y = rectangle_midpoint(p)
        plt.text(x, y, text, fontsize=8)
    return 0


def plot_rectangles(rectangles, names=True):
    """rectangles can be a dict or a list of rectangles. If rectangles is
a dict and names==True, the keys are usesd as names. names may also be
a list"""

    if isinstance(rectangles, dict):
        pairs = [(key, rectangles[key]) for key in list(rectangles)]
        rectangles = [x[1] for x in pairs]
        if names is True:
            names = [x[0] for x in pairs]
    not_plottable = 0
    for i, r in enumerate(rectangles):
        not_plottable += plot_rectangle(r,
                                        text=None if not names else names[i])
    if not_plottable:
        print('%d non-plottable rectangle-functions' % not_plottable)
    # plt.axis('equal')
    # plt.axis('equal') gives me problems when I forget to unset it for later plots
    # (with axis('auto')), so:
    if any([callable(x) for x in rectangles]):
        return
    xx = [p[0] for p in r for r in rectangles]
    yy = [p[1] for p in r for r in rectangles]
    xlims = [min(xx), max(xx)]
    ylims = [min(yy), max(yy)]
    xd = max(xlims) - min(xlims)
    yd = max(ylims) - min(ylims)
    if xd > yd:
        ycenter = (ylims[0] + ylims[1]) / 2
        ylims = [ycenter - xd / 2, ycenter + xd / 2]
    else:
        xcenter = (xlims[0] + xlims[1]) / 2
        xlims = [xcenter - yd / 2, xcenter + yd / 2]
    plt.plot(xlims, ylims, 'w.')


def rectangle_midpoint(p):
    return np.mean(np.array(p), axis=0)


# def combine_adjacent_rectangles_of_equal_size(rectangle_list):
#     ### ummm I'll take the midpoint of all... no. The two furthest
#     ### points, then the owwww. Find smallest rectangle that covers
#     ### all points. Google. Convex hull, of course. Generalize? Later
#     r = np.concatenate([np.array(r) for r in rectangle_list], axis=1)
#     #return convex_hull(r.transpose()) no, that's too sensitive

def combine_adjacent_rectangles_of_equal_size(rectangle_list):
    # in case they are not arrays
    rectangle_list = [np.array(r) for r in rectangle_list]
    # [[x,y], [x, y], ....]
    points = np.concatenate(rectangle_list, axis=1).transpose()
    midpoint = points.mean(axis=0)
    dists = [[np.linalg.norm(p - midpoint), i] for i, p in enumerate(points)]
    indexes = [x[1] for x in sorted(dists[:4], reverse=True)]
    # got the four points, now I have to make sure they don't cross
    return np.array(convex_hull(points[indexes])[:-1]).transpose()


def convex_hull(points):
    """from Mike Loukides at
    https://www.oreilly.com/ideas/an-elegant-solution-to-the-convex-hull-problem

    """
    def split(u, v, points):
        # return points on left side of UV
        return [p for p in points if np.cross(p - u, v - u) < 0]

    def extend(u, v, points):
        if not points:
            return []

        # find furthest point W, and split search to WV, UW
        w = min(points, key=lambda p: np.cross(p - u, v - u))
        p1, p2 = split(w, v, points), split(u, w, points)
        return extend(w, v, p1) + [w] + extend(u, w, p2)

    # find two hull points, U, V, and split to left and right search
    u = min(points, key=lambda p: p[0])
    v = max(points, key=lambda p: p[0])
    left, right = split(u, v, points), split(v, u, points)

    # find convex hull on each side
    return [v] + extend(u, v, left) + [u] + extend(v, u, right) + [v]


def point_inside_polygon(x, y, poly):
    # http://www.ariel.com.au/a/python-point-int-poly.html
    # determine if a point is inside a given polygon or not
    # Polygon is a list of (x,y) pairs.
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
