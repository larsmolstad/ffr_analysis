import os
import time
import pylab as plt
import numpy as np
from polygon_utils_old import plot_rectangles
import resdir

# todo maybe add a button instead of the while loop
def _ginput_show_info(df, fun=None, x='x', y='y'):
    print('Locate the plot window, click on a dot, or double-click (or triple-click) to quit')
    double_click_time = 0.5
    minimum_distance_index = None
    t0 = time.time()
    while 1:
        xy = plt.ginput(1, 0)
        xy = xy[0]
        previous_one = minimum_distance_index
        if xy[0] is None or time.time() - t0 < double_click_time:
            break
        distances = np.sqrt((df[x] - xy[0])**2 + (df[y] - xy[1])**2)
        minimum_distance_index = distances.argmin()
        print('\n\n')
        print(df.loc[minimum_distance_index])
        if fun:
            fun(df.loc[minimum_distance_index])
        t0 = time.time()
    print('Done. Setting plots back to inline. (Do "%matplotlib auto" to keep plotting in the external window)')
    get_ipython().magic('matplotlib inline')
    plt.gca().set_title('Done')
    return df.loc[previous_one] if previous_one else None


def _show_reg_fun(regr):
    def fun(df_row):
        plt.subplot(1, 2, 2)
        plt.cla()
        filename = os.path.join(resdir.raw_data_path,
                                df_row['filename'])
        regr.find_all_slopes(filename, do_plot=True)
        plt.gca().set_title(os.path.split(filename)[1])
    return fun


def ginput_check_points(df, rectangles, regr):
    get_ipython().magic('matplotlib auto')
    time.sleep(3)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.gca().axis('equal')
    plt.gca().set_title('triple click to quit')
    plot_rectangles(rectangles, names=True)
    plot_numbers = sorted(set(df.plot_nr))
    for nr in plot_numbers:
        d = df[df.plot_nr == nr]
        plt.plot(d.x, d.y, '.')
    return _ginput_show_info(df, _show_reg_fun(regr))


# Kind of the same, but plotting the slopes in the upper subplot


def ginput_check_regres(df, regr):
    get_ipython().magic('matplotlib auto')
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.gca().set_title('triple click to quit')
    plt.plot(df.t, df.N2O_slope, '.')
    return _ginput_show_info(df, _show_reg_fun(regr), x='t', y='N2O_slope')
