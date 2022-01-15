import numpy as np
import polygon_utils_old as polygon_utils


xmin = 599211.37
ymin = 6615133.59
angle = 0.7376


main_rectangle = polygon_utils.make_rectangle_with_angle(
    45.5 + xmin, -11.5 + ymin, 86.5, 83, angle)


def rectangles():
    # nw = northwest etc
    # pne is closest to the buildings
    pnw, pne, psw = (np.array(main_rectangle).transpose()
                     [i] for i in [3, 2, 0])
    # unit vectors
    une = (pne - pnw) / np.linalg.norm(pne - pnw)
    use = (psw - pnw) / np.linalg.norm(psw - pnw)

    def local_to_p(points):
        return [list(pnw + une * p[0] + use * p[1]) for p in points]
    plot_length = np.linalg.norm(pne - pnw) / 32
    width_limits = ([0, 10.8], [10.8, 21.6],
                    [27, 37.8], [37.8, 48.6],
                    [54, 64.8], [64.8, 75.6])
#    print('using fake rectangle width, see E22.py')
#    width_limits = ([0, 10.8], [10.8, 21.6+2],
#                    [27-2, 37.8], [37.8, 48.6+2],
#                    [54-2, 64.8], [64.8, 75.6])
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
