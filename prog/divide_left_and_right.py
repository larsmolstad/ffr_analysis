import numpy as np
from collections import defaultdict


nr_side = {1: 'left', 2: 'right'}


def search_sorted(v, x):
    # finds the I that minimizes abx(v[I]-x)
    def dist(a, b):
        return abs(a - b)
    v = np.array(v)
    I = v.searchsorted(x)
    if I < len(v) - 1 and abs(v[I + 1] - x) < abs(v[I] - x):
        I += 1
    if I == len(v):
        I -= 1
    return I


def find_shift_times(data_dict):
    """ returns i.e., [[t1, 'left'], [t2, 'right'] ...] """

    def check_alternating_1_0(a):
        if not (a[0] == 1 and
                all([x in [1, 0] for x in a]) and
                all([a[i] != a[i + 1] for i in range(len(a) - 1)])):
            print(a)
            raise Exception('should be alternating 1, 0, 1, 0 ...')

    def check_abbaabb_etc(a):
        if not (a[0] != a[1] and
                a[-1] != a[-2] and
                all([a[i] == a[i + 1] for i in range(1, len(a) - 1, 2)])):
            print(a)
            raise Exception('should be e.g., [1,2,2,1,1,2,2,1,1...]')

    def find_shift_times_old_data(data_dict):
        # used to start right always, didn't save the side
        t = [x[0] - data_dict['aux']['t'] for x in data_dict['side']]
        t = [ti for ti in t if ti >= 0]  # stoppet ikke alltid veksleren
        if len(t) > 1:
            t = [(t[i] + t[i + 1]) / 2 for i in range(0, len(t), 2)]
        sides = ['right'] if t else []
        for i in t[1:]:
            sides.append({'right': 'left', 'left': 'right'}[sides[-1]])
        return list(zip(t, sides))

    ts = data_dict['side']
    if ts and isinstance(ts[0][1], int):
        return find_shift_times_old_data(data_dict)
    t = [x[0] - data_dict['aux']['t'] for x in ts]
    s = [x[1] for x in ts]
    side, state = list(zip(*s))
    # a shift is characterized by opening one valve then closing the
    # other less than one second later (or maybe two, we'll see if I
    # change it)
    min_dt = 2
    d = np.diff(np.array(t))
    if not (np.all(d[::2] < min_dt) and
            np.all(d[1::2] > min_dt)):
        s = ('switching time diffs: {}' +
             '\n the first and then every other time shift should be' +
             '< 1sec').format(d)
        raise Exception(s)
    check_alternating_1_0(state)
    check_abbaabb_etc(side)
    t = [(t[i] + t[i + 1]) / 2 for i in range(0, len(t) - 1, 2)]
    return list(zip(t, [nr_side[x] for x in side[0::2]]))


def group(t, y, shift_t, cut_beginnings, cut_ends):
    """ returns {'left':(t,y,Istartstop),
                 'right':(t,y,Istartstop)}
        Istartstop is [(Istart, Istop),...] """
    ty = defaultdict(lambda: ([], [], []))
    t = np.array(t)
    times = [x[0] for x in shift_t] + [t[-1]]
    # todo bedre. tar med t[-1] i tilfelle det skjer et skifte rett
    # etter der.
    for i, (ts, side) in enumerate(shift_t):
        I1 = search_sorted(t, ts + cut_beginnings)
        I2 = search_sorted(t, times[i + 1] - cut_ends) + 1
        ty[side][0].extend(list(t[I1:I2]))
        ty[side][1].extend(list(y[I1:I2]))
        ty[side][2].append((I1, I2))
    return ty


def group_all(data_dict, cut_beginnings=3, cut_ends=2):
    shift_times = find_shift_times(data_dict)
    a = dict()
    for key, item in data_dict.items():
        if key in ['aux', 'side', 'Wind', 'filename']:
            continue
        a[key] = group(item[0], item[1], shift_times, cut_beginnings, cut_ends)
    return a
