import os
import sys
import time
import math
import numpy as np
import cPickle
import argparse
sys.path.append(os.path.split(os.path.split(
    os.path.realpath(__file__))[0])[0])  # sorry

import regression
import licor_indexes
import dlt_indexes
import get_data
import divide_left_and_right


class G:  # (G for global)
    res_file_name = 'slopes.txt'
    directory = ''
    regression_time = 100
    co2_guides = True
    co2_lag_time = 0
    startdate = False
    stopdate = False
    filter_fun = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finding slopes')
    parser.add_argument('directory', type=str, default=G.directory,
                        help='')
    parser.add_argument('-t', '--regression-time', type=float,
                        default=G.regression_time,
                        help='Length of regression segment, seconds. Default 100')
    parser.add_argument('--co2_guides', type=bool, default=True,
                        help='If True (default), CO2 determines the interval used for N2O regression')
    parser.add_argument('--startdate', type=str, default=False,
                        help='YYYY-MM-DD example: 2015-01-25')
    parser.add_argument('--stopdate', type=str, default=False,
                        help='YYYY-MM-DD example: 2015-01-25')
    parser.add_argument('--filter_fun', type=str, default=False,
                        help='Name of a python file and function filtering which raw data files to use. See myfilter.py for an example. Example: --filter_fun myfilter.filter_fun')
    parser.add_argument('--out', type=str, default=G.res_file_name,
                        help='Result file. Default slopes.txt')
    args = parser.parse_args()
    G.directory = args.directory
    G.regression_time = args.regression_time
    G.co2_guides = args.co2_guides
    G.startdate = args.startdate
    G.stopdate = args.stopdate
    G.res_file_name = args.out
    if args.filter_fun:
        import importlib
        filename, function_name = args.filter_fun.split('.')
        G.filter_fun = getattr(importlib(filename), function_name)


def plot_regressions(data, regressions, plotfun, normalized=True, do_plot=True):
    """ plotting the n2o and co2 with regression lines. 
    The CO2 is scaled to match the N2O"""
    def min_and_max(x): return min(x), max(x)

    def stretch(y, a, b):
        return [(x + b) * a for x in y]

    def make_plotargs(key, color, a, b):
        plotargs = []
        d = data[key]

        def nrm(y):
            return stretch(y, a, b)
        plotargs.extend([d[0], nrm(d[1]), color + '.'])
        for side, rdict in regressions.iteritems():
            reg = rdict[key]
            t0, y0 = data[key][:2]
            t = [ti for (I1, I2) in reg.Iswitch for ti in t0[I1:I2]]
            y = [yi for (I1, I2) in reg.Iswitch for yi in y0[I1:I2]]
            t = t[reg.start:reg.stop]
            y = y[reg.start:reg.stop]
            plotargs.extend([t, nrm(y), color + 'o',
                             t, nrm(reg.intercept + reg.slope * np.array(t)),
                             color])
        return plotargs
    plotargs = []
    min_n2o, max_n2o = min_and_max(data['N2O'][1])
    min_co2, max_co2 = min_and_max(data['CO2'][1])
    a = (max_n2o - min_n2o) / max(1e-10, max_co2 - min_co2)
    b = (min_n2o * max_co2 - max_n2o * min_co2) / max(1e-10, max_n2o - min_n2o)
    plotargs.extend(make_plotargs('CO2', 'b', a, b))
    plotargs.extend(make_plotargs('N2O', 'r', 1, 0))
    ymin = min_n2o
    ymax = max_n2o
    span = ymax - ymin
    if do_plot:
        plotfun(*plotargs)
    else:
        return plotargs
        #plotfun('set_ylim', ymin-span*.1, ymax+span*.1)


def remove_zeros(t, y):
    t = np.array(t)
    y = np.array(y)
    t = t[y != 0]
    y = y[y != 0]
    return t, y


def find_all_slopes(filename_or_data, interval, co2_guides=True,
                    plotfun=None):
    """Finds the regression lines for N2O and CO2, for left and right
    side. The parameter "interval" is the width in seconds of the
    segment over which to perform the regressions.  If
    co2_guides==True, the segment where the CO2 curve is
    steepest is used for the N2O regressions.
    
    returns {'left':{'CO2':(Regression, (x,y)),'N2O':...}, {'right': ...}}

    """
    if isinstance(filename_or_data, str):
        data = get_data.get_file_data(filename_or_data)
    else:
        data = filename_or_data
    resdict = divide_left_and_right.group_all(data)
    regressions = {'left': {}, 'right': {}}
    keys = 'CO2 N2O'.split()
    for side in regressions.keys():
        tbest = None
        for key in 'CO2 N2O'.split():
            t, y = remove_zeros(*resdict[key][side][:2])
            Iswitch = resdict[key][side][2]
            if key != 'CO2' and co2_guides and tbest is not None:
                a = regression.regress_within(t, y, *tbest)
            else:
                a = regression.find_best_regression(t, y, interval,
                                                    'steepest')
                if key == 'CO2' and a is not None:
                    tbest = t[a.start], t[a.stop]
            if a is not None:
                a.Iswitch = Iswitch
                regressions[side][key] = a
        if len(regressions[side].keys()) == 0:
            regressions.pop(side)
    if plotfun is not None:
        plot_regressions(data, regressions, plotfun)
    return regressions


def write_result_to_file(res, name, f):
    for side, sideres in res.iteritems():
        s = os.path.split(name)[1] + '\t' + side
        for key, regres in sideres.iteritems():
            s += '\t{0}\t{1}'.format(key, regres.slope)
        print s
        f.write(s + '\n')


def find_regressions(directory, res_file_name, regression_time):
    files = get_data.select_files(directory, G)
    n = len(files)
    i = 0
    t0 = time.time()
    resdict = {}
    with open(res_file_name, 'w') as f:
        for name in files:
            t = time.time()
            if t - t0 > 0.5:
                print('%d/%d' % (i, n))
                t0 = t
            try:
                data = get_data.get_file_data(name)
                # res = find_fluxes(data,regression_time)
                res = find_all_slopes(data, regression_time, G.co2_guides)
                write_result_to_file(res, name, f)
                resdict[os.path.split(name)[1]] = res
            except Exception, e:
                import traceback
                traceback.print_exc()
                print 'continuing'
                continue
            i += 1
    with open(os.path.splitext(res_file_name)[0] + '.pickle', 'w') as f:
        cPickle.dump(resdict, f)


def print_reg(regres):
    for k, dct in regres.iteritems():
        print k
        for subst, reg in dct.iteritems():
            print subst
            print reg


if __name__ == '__main__':
    find_regressions(G.directory, G.res_file_name, G.regression_time)
