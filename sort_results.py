"""
"""
# todo a lot of old code here, mostly from the time before I started using Pandas

import re
import os
import time
import xlwt
import tkinter.messagebox
import tkinter.filedialog
import sys
import datetime
from collections import defaultdict
import argparse
from math import sin, cos, pi

import last_directory
from migmin import migmin_rectangles
import find_plot
from get_data import number_after, parse_filename
import numpy as np
import pandas as pd
import scipy.integrate
import resdir
from plotting_compat import plt
plot = plt.plot


class G:  # (for global)
    slope_file = os.path.join(resdir.slopes_path, 'slopes.txt')  # 'repeat'
    xls_file = 'results.xls'
    chamber_distance = 2  # meters, sideways distance from gps to chamber center
    chamber_fw_distance = 0.2  # forward distance from gps to chamber center


description = """
Sorts the slopes according to position and writes the results to excel
Examples
python sort_results.py
python sort_results.py -s slopes.txt 
python sort_results.py -s ..\slopes.txt 
python sort_results.py -s repeat --out resultfile.xls
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--slope_file', type=str, default='',
                        help='If not given, a dialog box opens. \nIf repeat, it tries to use the same as last time')
    parser.add_argument('--out', type=str, default='results.xls',
                        help='Result file. Default results.xls')
    args = parser.parse_args()
    G.slope_file = args.slope_file
    G.xls_file = args.out

lastfile = '.sort_result_lastfile'
last_wp_file = '.sort_results_wp_file'


def getfile(remember_file, title="open"):
    lastdir = last_directory.remember(remember_file)
    file = tkinter.filedialog.askopenfilename(initialdir=os.path.split(lastdir.get())[0],
                                        title=title)
    lastdir.set(file)  # os.path.split(file)[0])
    return file


if G.slope_file == 'repeat':
    G.slope_file = last_directory.remember(lastfile).get()
elif not G.slope_file:
    G.slope_file = getfile(lastfile, "Select slope file")


def chamber_position(vehicle_pos, side):
    x, y, heading = vehicle_pos['x'], vehicle_pos['y'], vehicle_pos['heading']
    d = G.chamber_distance
    dfw = G.chamber_fw_distance
    if heading == float('nan') or np.isnan(heading):
        return (x, y)
    else:
        ang = {'left': heading + pi / 2, 'right': heading - pi / 2}[side]
        # ang = {'left':pi - heading, 'right':heading}[side]
        return (x + d * cos(ang) + dfw * np.cos(heading),
                y + d * sin(ang) + dfw * np.sin(heading))


def all_positions(filenames, sides):
    q = list(zip([parse_filename(s)['vehicle_pos'] for s in filenames], sides))
    return [chamber_position(vehicle_pos, side)
            for vehicle_pos, side in q]


def dictify(res):
    # res is like [name, side,'CO',number,'CO2',number,'N2O',number]
    # {res[i]:res[i+1] for i in range(1,len(res),2)}
    # to better find errors in the result files:
    y = {}
    for i in range(2, len(res), 2):
        try:
            y[res[i]] = res[i + 1]
        except Exception as e:
            try:
                print('\nres[i], res[i+1] = \n', res[i], res[i + 1])
            except:
                pass
            print('\n')
            raise(e)
    return y, res[1]


def make_df(raw_result_list):
    filenames = [x[0] for x in raw_result_list]
    sides = [x[1] for x in raw_result_list]
    p = all_positions(filenames, sides)
    y = []
    for i, name in enumerate(filenames):
        rdict = parse_filename(name)
        slopes, rdict['side'] = dictify(raw_result_list[i])
        for key, val in slopes.items():
            rdict[key] = val
        pos = rdict['vehicle_pos']
        xy = chamber_position(pos, rdict['side'])
        # rdict['chamber_pos'] = {'x':xy[0], 'y':xy[1]}
        rdict['x'], rdict['y'] = xy
        rdict['vehicle_x'] = pos['x']
        rdict['vehicle_y'] = pos['y']
        rdict['vehicle_z'] = pos['z']
        rdict['heading'] = pos['heading']
        rdict['used_sides'] = pos['side']
        rdict['daynr'] = np.floor(rdict['t'] / 86400)
        y.append(rdict)
    df = pd.DataFrame(y).drop('vehicle_pos', axis=1)
    #df['id'] = df.index
    return df


def simplify_df(df):
    todrop = ['name', 'side', 'vehicle_pos', 'vehicle_x',
              'vehicle_y', 'vehicle_z', 'used_sides']
    return df.drop(todrop, axis=1, errors='ignore')


def rearrange_df(df):
    tomove = ['name', 'side', 'vehicle_pos', 'vehicle_x',
              'vehicle_y', 'vehicle_z', 'used_sides']
    tomove = [x for x in tomove if x in df.columns]
    tokeep = [x for x in df.columns if x not in tomove]
    return df[tokeep + tomove] #df.drop(todrop, axis=1, errors='ignore')

def make_list_with_parsed_filenames(raw_result_list):
    filenames = [x[0] for x in raw_result_list]
    sides = [x[1] for x in raw_result_list]
    p = all_positions(filenames, sides)
    y = []
    for i, name in enumerate(filenames):
        rdict = parse_filename(name)
        rdict['slopes'], rdict['side'] = dictify(raw_result_list[i])
        xy = chamber_position(rdict['vehicle_pos'], rdict['side'])
        rdict['chamber_pos'] = {'x': xy[0], 'y': xy[1]}
        y.append(rdict)
    return y


def group_results(raw_result_list, rectangles):
    rdl = make_list_with_parsed_filenames(results)
    for rd in rdl:
        x, y = rd['x'], rd['y']
        rd['plot_nr'] = find_plot.find_plot(x, y, rectangles)
    return None


def assign_plots(df, rectangles):
    df['plot_nr'] = find_plot.find_plots(df.x, df.y, rectangles)


def list2list_dict(lst, key='plot_nr'):
    """ lst is a list of dicts which all have key (default 'plot_nr').
     The items of lst are returned, but divided according to that key in a dict of lists.
    This is almost the inverse of what res_list does"""
    y = defaultdict(list)
    for x in lst:
        y[x[key]].append(x)
    return dict(y)


def get_result_list_from_slope_file(slope_file,
                                    start=0,
                                    stop=-1,
                                    index_list=None):
    """ returns reslist
        reslist[i] is like [filename, side, name, number, name, number ...]
    """
    def str2num_line(s):
        # reslist[i] is like [filename, side, name, number, name, number ...]
        # this converts the numbers from strings to floats: (in place)
        for i in range(3, len(s), 2):
            s[i] = float(s[i])
    with open(slope_file) as f:
        a = f.readlines()
    a = [x.strip('\n\r') for x in a]
    reslist = [x.split() for x in a if x]
    reslist = reslist[start:stop] if stop>=0 else reslist[start:]
    if index_list is not None:
        reslist = [x[i] for x in index_list]
    for s in reslist:
        str2num_line(s)
    return reslist


def make_df_from_slope_file(name,
                            rectangles,
                            treatment_dict,
                            remove_redoings_time=3600):
    unsorted_res = get_result_list_from_slope_file(name, start=0)
    df0 = make_df(unsorted_res)
    assign_plots(df0, rectangles)
    df0['treatment'] = df0.plot_nr.map(lambda x: treatment_dict[x]
                                       if x in treatment_dict else None)
    df = rearrange_df(df0[df0.plot_nr > 0])
    if remove_redoings_time:
        df = remove_redoings(df, remove_redoings_time)
    return df, df0


def res_list(resdict):
    """ 
    "flattening" the dict to a list,
    as a side-effect, adds a 'plot_nr' key to each measurement in res
    res is a dict of lists
    """
    y = []
    for plotnr, plotres in resdict.items():
        for p in plotres:
            x = p
            x['plot_nr'] = plotnr
            y.append(x)
    return y


resdict2list = res_list


def transform_resdict(fun, resdict):
    """ fun takes two arguments, the slope and (used optionally) 
the whole result dict for one measurement"""
    import copy
    lst = res_list(copy.deepcopy(resdict))
    #[x.copy() for x in res_list(resdict)]
    for x in lst:
        x['slopes']['N2O'] = fun(x['slopes']['N2O'], x)
    return list2list_dict(lst)


def filter_resdict(fun, resdict):
    """ 
    picking all res for which fun is true
    same as {key:[x for x in lst if fun(x)] for key, lst in resdict.iteritems()}
    """
    res2 = {}
    for key, reslist in resdict.items():
        y = [r for r in reslist if fun(r)]
        res2[key] = y
    return res2


def find_new_pos(p0, p):
    def closest(p0, p, yfak):
        x0, y0 = p0
        x, y = list(zip(*p))
        d = (x - x0)**2 + (yfak * (y - y0))**2
        return min(d)
    while len(p) and closest(p0, p, 1) < 10:
        p0[0] += 1
    return p0


def last_part_of_name(name):
    def find_last_substring(long_string, substring):
        i = -2
        while i != -1:
            j = i
            i = long_string.find(substring, max(i, 0) + 1)
        return max(j, -1)
    i = max(find_last_substring(name, 'left'),
            find_last_substring(name, 'right'))
    if i == -1:
        i = 10
    return name[i:].replace('_Plot_', '').replace('_', '')


# we want to have empty cells for missing measurements in the
# spreadsheet.  problem with that: If the same plot is measured twice
# in a day it might be because all plots are measured twice or because
# the robot was stopped and restarted. Anyway I will have to sort them
# by days and make a rule for what to keep and what to insert as empty.
# Must go through the measurements day by day instead of plot by plot


def xlswrite_from_df(name, df, do_open=False):
    #daynrs = sorted(set(df.daynr))# but sometimes we measure twice per day
    workbook = xlwt.Workbook()
    date_format = xlwt.XFStyle()
    date_format.num_format_str = 'dd/mm/yyyy'
    treatments = sorted(set(df.treatment))
    for compound in ['N2O', 'CO2']:# todo name (not in df so need df0) and x and y
        w = workbook.add_sheet(compound)
        i = 0
        for treatment in treatments:
            d = df[df.treatment==treatment]
            plots = sorted(set(d.plot_nr))
            for j, plot_nr  in enumerate(plots):
                i += 2
                w.write(0, i-1, treatment)
                w.write(1, i-1, str(plot_nr))
                d2 = d[d.plot_nr==plot_nr]
                t = d2.t.values
                y = d2[compound].values
                for rownr, ti in enumerate(t):# todo vectorize?
                    ti = datetime.datetime.utcfromtimestamp(ti)
                    w.write(rownr+2, i-1, ti, date_format)
                    w.write(rownr+2, i, y[rownr])
        # try:
        #     workbook.save(name)
        # except IOError:
        #     tkMessageBox.showinfo(
        #         "Close the old xls-file, then press ok...........")
    try:
        workbook.save(name)
    except IOError:
        raise IOError("You must close the old xls file")
    if do_open:
        os.startfile(name)


def get_tvxy(resdict, plotnr, substance, tmin=0, tmax=1e99):
    ty = [(x['t'], float(x['slopes'][substance]), x['x'], x['y'])
          for x in sorted_results[plotnr] if tmin < x['t'] < tmax]
    return list(zip(*ty))


# def test(start=0, stop=1e99, cumulative=False):
#     # maa ha lagd ax og ax2 forst todo rette opp ax og ax2
#     ax.hold(False)
#     ax2.hold(False)
#     time.sleep(.1)
#     ax2.axis('equal')
#     time.sleep(.1)
#     colors = 'krgbcy'
#     annpos = []
#     for i, c in enumerate(colors):
#         for j in find_plot.get_treatment_plots('CDLMNO'[i]):
#             t, v, x, y = get_tvxy(s, j, 'N2O', start, stop)
#             t = (np.array(t) - time.mktime((2015, 1, 1, 0, 0, 0, 0, 0, 0))) / 86400
#             if cumulative:
#                 v = scipy.integrate.cumtrapz(v, t) * 86400
#                 v = np.concatenate((np.array([0]), v))
#             ax.plot(t, v, c + '.-', {'markersize': 10});
#             ax2.plot(x, y, c + '.', {'markersize': 10})
#             tc()
#             if cumulative:
#                 annpos.append(find_new_pos([t[-1] + 1, v[-1]], annpos))
#                 ax.text(annpos[-1][0], annpos[-1][1],
#                         repr(j) + ':' + 'CDLMNO'[i], {'fontsize': 8})
#                 tc()
#             if i == 0:
#                 ax.plot('hold', True)
#                 a2.plot('hold', True)
#                 tc()


# write_treatments(find_plot.plots_utm, find_plot.treatments, do_show_plots=False,ax=ax)
# test(cumulative = True,start = time.time()-86400*60)


# def show_pos():
#     filenames = [x[0] for x in results]
#     sides = [x[1] for x in results]
#     p = all_positions(filenames, sides)
#     p2 = list(set(p))
#     x = [q[0] for q in p2]
#     y = [q[1] for q in p2]
#     plot([xx - x[0] for xx in x],y,'.')

def show_pos2(results, start, stop, dt=.5, pr=False):
    if isinstance(results[0], list):  # for unsorted results
        filenames = [x[0] for x in results]
        sides = [x[1] for x in results]
    else:  # for sorted results
        filenames = [x['name'] for x in results]
        sides = [x['side'] for x in results]
    p = all_positions(filenames, sides)
    #    p2 = list(set(p))
    x = [q[0] for q in p]
    y = [q[1] for q in p]
    plot('axis', 'equal')
    plot(x, y, 'y.')
    plot('hold', True)
    plot(x[start], y[start], 'k.')
    if pr:
        print(filenames[start])
    for i in range(start + 1, stop):
        if pr:
            print(filenames[i])
        plot(x[i - 1], y[i - 1], 'y.', x[i], y[i], 'k.')
        time.sleep(dt)
    plot('hold', False)


def show_plots():
    x = [p[0] for p in pos]
    y = [p[1] for p in pos]
    plot(x, y, '.')
    for i in range(len(x)):
        plot('text', x[i], y[i], i)


# if __name__ == '__main__' and G.xls_file not in ['False', 'None']:
#     temp = get_results_from_slope_file(G.slope_file, rectangles='default')
#     results, sorted_results, resdict = temp
#     xlswrite(G.xls_file, sorted_results, True)



def plot_points_in_rectangles(results, rectangles='default',
                              anim_start=0, stop=0, dt=.5, pr=False):
    import plot_rectangles
    plt.cla()
    plt.hold(True)
    if rectangles == 'default':
        rectangles = plot_rectangles.migmin_rectangles()
    plot_rectangles.plot_rectangles(rectangles)
    show_pos2(results, anim_start, stop, dt=dt, pr=pr)


# if __name__ == '__main__':
#     unsorted_results, res, resdict = get_results_from_slope_file(
#         G.slope_file, start=-20)
#     show_pos2(unsorted_results, 1, 10, dt=.5, pr=False)
    # print "kommenterte ut if __name__ == '__main__'"


def pick_data(res, plotnr, name='N2O'):
    return list(zip(*[(p['t'], p['slopes'][name]) for p in res[plotnr]]))


# todo fjerne redundans, bruke pick_data. f

def plot_plots(resdict, plots, symbol='.', markersize=3, ret=None):
    a = []
    for key in plots:
        r = resdict[key]
        x = []
        y = []
        for p in r:
            x.append((p['t'] - 0 * resdict[plots[0]][0]['t']) / 86400)
            y.append(p['slopes']['N2O'])
        print(('%f' % min(y)))
        a.append(x)
        a.append(y)
        a.append(symbol)
    plot(*a, markersize=markersize)
    return a if ret is not None else None


def plot_plots(df, plots, symbol='.', markersize=3, ret=None):
    a = []
    for key in plots:
        r = df[df.plot_nr == key]
        a.append(r.t / 86400)
        a.append(r.N2O)
        a.append(symbol)
    plot(*a, markersize=markersize)
    return a if ret is not None else None


order = '_mcndclmonodllonmmocnddlc'
order = '_nolomndcdlnmocdlomlcdncm'
treatments = {i: order[i] for i in range(1, len(order))}


def find_all(a):
    I = []
    for i in range(len(order)):
        if order[i] == a:
            I.append(i)
    return I


def cla():
    plot('cla')


def plot_treatment(c, resdict, ret=False):
    I = find_all(c)
    return plot_plots(resdict, I, '.', ret=ret)  # if c=='c' else c)


def control(resdict):
    plot_treatment('c', resdict)


def marble(resdict):
    plot_treatment('m', resdict)


def norite(resdict):
    plot_treatment('n', resdict)


def dolomite(resdict):
    plot_treatment('d', resdict)


def larvikite(resdict):
    plot_treatment('l', resdict)


def olivine(resdict):
    plot_treatment('o', resdict)


def run_through_all(resdict):
    means = []
    s = 'cmndlo'
    for c in s:
        print(c)
        a = plot_treatment(c, resdict, ret=True)
        y = a[1::3]
        means.append([np.mean(x) for x in y])
    return s, means


def get_resdict_means_dict(resdict):
    return average_something(resdict, ['slopes', 'N2O'])


def get_resdict_means(resdict, treatments='cmndlo'):
    """returns treatments, means"""
    means = []
    s = treatments
    for c in s:
        plot_nrs = find_all(c)
        y = []
        for key in plot_nrs:
            y.append([p['slopes']['N2O'] for p in resdict[key]])
        means.append([np.mean(x) for x in y])
    return s, means


def compare(letters, resdict):
    cla()
    time.sleep(1)
    symbols = '.*osh+xd'
    for i, c in enumerate(letters):
        plot_plots(resdict, find_all(c), symbols[i])
        time.sleep(1)


def plot_points(resdict, nrs):
    xy = []
    for i in nrs:
        xy.append([r['x'] for r in resdict[i]])
        xy.append([r['y'] for r in resdict[i]])
        xy.append('*')
    plot('cla')
    time.sleep(.1)
    plot(*xy)
    return xy


def minimum_coords(resdict):
    xmin = 1e99
    ymin = 1e99
    for i in resdict:
        xmin = min(xmin, min([r['x'] for r in resdict[i]]))
        ymin = min(ymin, min([r['y'] for r in resdict[i]]))
    return xmin, ymin


def simplified_coords(resdict, subtract='min'):
    xmin, ymin = minimum_coords(resdict) if subtract == 'min' else subtract
    r = {}
    for i in resdict:
        r[i] = [[x - xmin for x in [q['x'] for q in resdict[i]]],
                [y - ymin for y in [q['y'] for q in resdict[i]]]]
    return r


def pl(resdict, nrs, starnr):
    import copy
    q = simplified_coords(resdict, [0, 0])
    qq = []
    for i in nrs:
        qq.extend([q[i][0], q[i][1], 'y.'])
    if starnr:
        qq.extend([q[starnr][0], q[starnr][1], '*'])
    plot('cla')
    time.sleep(.1)
    plot(*qq)


def pl2(resdict, nrs):
    plot('cla')
    time.sleep(5)
    for i in nrs:
        pl(resdict, nrs, i)
        time.sleep(2)


def numbers(resdict):
    pl(resdict, list(range(1, 25)), [])
    for i, p in enumerate(pos):
        time.sleep(1)
        plot('text', p[0], p[1], repr(i + 1))


def ginput_and_limits(plot):
    g = None
    while g is None:
        g = plot.ginput()
    xy = g[0]
    ylim = plot.request('get_ylim')
    xlim = plot.request('get_xlim')
    return xy, xlim, ylim


def ginput_find_closest(xydata, plot):
    # xydata er [[xdata, ydata], [xdata, ydata]]
    # returnerer I, J
    xy, xlim, ylim = ginput_and_limits(plot)
    if re.search('button=(\d)', xy[2]).group(1) == '3':
        return -1, -1
    xlim = max(xlim) - min(xlim)
    ylim = max(ylim) - min(ylim)
    dist = None
    best = (0, 0)
    for i, (xdata, ydata) in enumerate(xydata):
        for j, (x, y) in enumerate(zip(xdata, ydata)):
            newdist = ((xy[0] - x) / xlim)**2 + ((xy[1] - y) / ylim)**2
            if dist is None or dist > newdist:
                best = i, j
                dist = newdist
    return best


def get_xydata(plot):
    plot('get_lines')
    time.sleep(.2)
    q = plot.read_return()
    return [(w.get_xdata(), w.get_ydata()) for w in q]


def get_xys(df, plotnrs, name='N2O'):
    # df2 = df[df.plot_nr in plotnrs] # doesn't work
    # df2 = df[df.apply(lambda x:x.plot_nr in plotnrs, axis=1)] #slow
    #dflist = [df[df.plot_nr == i] for i in plotnrs]
    #df2 = pd.concat(dflist)
    v = []
    for n in plotnrs:
        df2 = df[df.plot_nr == plotnr]
        v.append(df2.t.values / 86400, df2[name].values)
    return v


def ginput_find_closest_res(df, plotnrs, plotfun=plt.plot):
    # plot_plots(res, plotnrs)
    df2 = df[df.plot_nr.isin(plotnrs)]
    a = [[df2.t.values / 86400, df2.N2O.values]]
    I, J = ginput_find_closest(a, plotfun)
    if I == -1:
        return False
    row = df2.iloc[J]
    return row


def ginput_show_regression(directory, df, plotnrs, plot_first=True,
                           plotfun=plt.plot, reg_plotfun=plt.plot):
    import find_regressions
    if isinstance(plotnrs, int):
        plotnrs = [plotnrs]
    #plot('subplot', 211)
    if plot_first > 0:
        print('plotting first...')
        print(plotnrs)
        plot_plots(df, plotnrs)
    print('now ginput')
    q = ginput_find_closest_res(df, plotnrs)
    if False:  # not q:
        return False
    else:
        if plot_first > -1:
            plotfun('cla')
            plot_plots(df, plotnrs)
        x, y = q.t / 86400, q.N2O
        plotfun('hold', True)
        plotfun(x, y, 'o', markerfacecolor='none')
        plotfun('hold', False)
        filename = os.path.join(directory, q['name'])
        #plot('subplot', 212)
        return find_regressions.find_all_slopes(filename,
                                                interval=100,
                                                co2_guides=True,
                                                plotfun=reg_plotfun)


def ginput_some_regressions(df, I, reg_plotfun=plt.plot):
    import traceback
    reg = True
    first = True
    n = 3
    while reg and n > 0:
        try:
            reg = ginput_show_regression(resdir.raw_data_path,
                                         df, I, first,
                                         reg_plotfun=reg_plotfun)
            first = False
        except:
            n -= 1
            print(traceback.format_exc())
            print('oj')


subplot_dict = {}


def subplot(i, j, k, find_pos=False):
    plt.plot('subplot', i, j, k)
    if find_pos:
        subplot_dict[(i, j, k)] = plt.plot.request('get_position')


def many_subplots(n, m):
    for i in range(n * m):
        subplot(n, m, i + 1)


def plot_some_regressions(resdict, nr, plots, n, m):
    import my_plotter as mp
    import get_data
    import find_regressions
    import math
    plt.clf()
    for i in plots:
        r = resdict[i][nr]  # todo...
        a = get_data.get_file_data(
            os.path.join(resdir.raw_data_path, r['name']))
        subplot(4, 4, i)
        reg = find_regressions.find_all_slopes(
            a, interval=100, co2_guides=True, plotfun=plt.plot)


def plot_some_regressions2(resdict, nr, plots, n, m):
    import my_plotter as mp
    import get_data
    import find_regressions
    import math
    plt.clf()
    directory = resdir.raw_data_path
    for i in plots:
        r = resdict[i][nr]  # todo...
        a = get_data.get_file_data(os.path.join(directory, r['name']))
        subplot(4, 4, i)
        reg = find_regressions.find_all_slopes(
            a, interval=100, co2_guides=True, plotfun=plt.plot)


def datestr2sec(s):
    """ seconds since epoch for datestring s 
Format of s is "%Y%m%d", "%Y%m%d %H", "%Y%m%d %H%M" or "%Y%m%d %H%M%S"
e.g.: "20170525 07"  is 7AM 25th of May 2017"""
    if isinstance(s, float):
        return s
    fmt = "%Y%m%d"
    fmt += {8: "", 11: " %H", 13: " %H%M", 15: " %H%M%S"}[len(s)]
    return time.mktime(time.strptime(s, fmt))


def filter_for_time(resdict, t0, t1):
    t0 = datestr2sec(t0)
    t1 = datestr2sec(t1)
    return filter_resdict(lambda x: t0 <= x['t'] < t1, resdict)


def t_to_daynr(t):
    return np.floor(t * 1.0 / 86400)


def find_average_slope_old(resdict, minimum_number_of_measurements_in_a_day=10):
    """ finds average slope versus day on the whole field"""
    y = defaultdict(list)
    for plotnr, reslist in resdict.items():
        if plotnr < 0:
            continue
        for r in reslist:
            y[t_to_daynr(r['t'])].append(r['slopes']['N2O'])
    ym = dict()
    for key, item in y.items():
        if len(item) > minimum_number_of_measurements_in_a_day:
            ym[key] = np.mean(item)
    return ym


def filter_for_average_slope_old(resdict, lower=0.0001, upper=np.inf):
    """makes a new resdict containing only the results from the days
    where the average slopes were between lower and upper"""
    y = find_average_slope(resdict)
    y = [(val, key) for (key, val) in y.items()]
    daynr = [int(x[1]) for x in y if x[0] > lower and x[0] < upper]
    res2 = filter_resdict(lambda r: (int(r['t'] / 86400) in daynr), resdict)
    return res2


def filter_for_average_slope_days(df, lower=0.0001, upper=np.inf):
    """ makes a new dataframe containing only the results from the days
    where the average slopes were between lower and upper"""
    means = df.groupby('daynr').N2O.mean()
    days = means[(means < upper) & (means >= lower)].index
    return df[df.daynr.isin(days)]


def find_nonlast_redoings(df, nr, dt=3600):
    """Finds where the robot has measured in the same plot (and on the
    same side) after less than dt (in seconds).  Ususally, if dt is
    small, this is due to something going wrong, and the measurement
    has been restarted.

    """
    d = df[df.plot_nr == nr]
    #sides = df0.loc[d.index].side
    d_left = d[d.side == 'left']
    d_right = d[d.side == 'right']
    left = np.where(np.diff(d_left.t) < dt)
    right = np.where(np.diff(d_right.t) < dt)
    # return left, right, d_left, d_right
    return pd.concat([d_left.iloc[left], d_right.iloc[right]])


def remove_redoings(df, dt=3600):
    """
    Returns a dataframe where measurements has been removed that has
    been redone within dt seconds. The intention is to remove
    measurements that have failed. 

    """
    plotnrs = set(df.plot_nr)
    to_remove = [find_nonlast_redoings(df, i)
                 for i in plotnrs]
    if to_remove:
        to_remove = pd.concat(to_remove)
        df.drop(to_remove.index)
        return df
    else:
        return df

#+END_SRC


class Indexes():
    def __init__(self):
        self.norite = [1, 6, 11, 22]
        self.olivine = [2, 4, 13, 17]
        self.larvikite = [3, 10, 16, 19]
        self.marble = [5, 12, 18, 24]
        self.dolomite = [7, 9, 15, 21]
        self.control = [8, 14, 20, 23]


material_plotnumbers = Indexes()
# for x in dir(sr.ind)[3:]:
#    print getattr(sr.ind, x) == sr.find_all(x[0])

# trenger x, y, fluks og behandling(farger)
# def make_selector_fun(fun):
#     if not callable(fun):
#         if isinstance(fun, str):
#             pass
#         if len(fun)==1:
#             a = fun[0]
#             fun = lambda x:x[a]
#         elif len(fun)==2:
#             a,b = fun
#             fun = lambda x:x[a][b]
#     return fun

# def collect_to_resdict(resdict, fun, keys=None):
#     fun = make_selector_fun(fun)
#     if keys is None:
#         keys = resdict.keys()
#     y = dict()
#     for key in keys:
#         y[key] = [fun(x) for x in resdict[key]]
#     return y


def average_something(resdict, fun):
    """ fun may be a function or lambda expression, which is then returned, 
or a string or a list of strings, which will be turned into a lambda expression 
of the type lambda x:x[fun] or lamdba x:x[fun[0]][fun[1]]"""
    if not callable(fun):
        if isinstance(fun, str):
            fun = [fun]
        if len(fun) == 1:
            a = fun[0]

            def fun(x): return x[a]
        elif len(fun) == 2:
            a, b = fun

            def fun(x): return x[a][b]
    y = dict()
    for key, item in resdict.items():
        x = [fun(x) for x in item if fun(x) is not None]
        y[key] = sum(x) / len(x) if len(x) else None
    return y


def plot_bars_old(x, y, z):
    for X, Y, Z in zip(x, y, z):
        plot('bar', [X], [Z], [Y], zdir='y', color='r', alpha=0.8)


def plot_bars(x, y, z, thickness, color, alpha=1):
    d = np.ones(len(x)) * thickness
    plot('bar3d', x, y, d * 0, d, d, z, color=color, alpha=alpha)


def plot_barmap(df, offsets=[0, 0], theta=0, thickness=4, alpha=1):
    """ makes a bar plot of the average results in each plot
    What about side, then. Later. See the .org-file"""
    import matplotlib.pyplot as plt
    g = df.groupby('plot_nr')
    # average_something(resdict, ['slopes', 'N2O'])
    average_slope = g.N2O.mean()
    average_x = g.x.mean()  # average_something(resdict,['chamber_pos', 'x'])
    average_y = g.y.mean()  # average_something(resdict,['chamber_pos', 'y'])
    treatment = g.treatment.first()
    keys = [x for x in treatment.index if x >= 0]
    # keys = [key for key in average_slope if average_slope[key] is not None]
    # keys.sort()
    x = np.array([average_x[key] for key in keys]) - offsets[0]
    y = np.array([average_y[key] for key in keys]) - offsets[1]
    x = np.cos(theta) * x - np.sin(theta) * y
    y = np.sin(theta) * x + np.cos(theta) * y
    z = [average_slope[key] for key in keys]
    colors = ['b'] * len(x)
    colordict = {'norite': 'r', 'olivine': 'g', 'larvikite': 'k',
                 'marble': 'w', 'dolomite': 'y', 'control': 'b'}
    for material, color in colordict.items():
        for i in getattr(material_plotnumbers, material):
            if i in keys:
                colors[keys.index(i)] = color
    plot_bars(x, y, z, thickness, color=colors, alpha=alpha)
    # make the legend
    matkeys = list(colordict.keys())
    proxies = [plt.Rectangle((0, 0), 1, 1, fc=colordict[material])
               for material in matkeys]
    names = [k[0] for k in matkeys]
    plot('legend', proxies, names)
    return x, y, z, keys, colors, proxies, names


def barmap_splitted(df, thickness=2, alpha=1, theta=np.pi / 4,
                    do_clf=True, ret=False):
    if do_clf:
        plt.clf()
        plt.subplot(111, projection='3d')
    # both_sides = df0.vehicle_pos.map(lambda x:x['side']=='both')[df.index]
    dfboth = df[df.used_sides == 'both']
    # todo
    heading = dfboth.heading
    side = dfboth.side
    nw = (side == 'left') != (heading > -1)
    se = ~nw
    x1, y1, z1, _, colors1, proxies, names = plot_barmap(
        dfboth[nw], theta=theta, thickness=thickness)
    x2, y2, z2, _, colors2, proxies, names = plot_barmap(
        dfboth[se], theta=theta, thickness=thickness)
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    z = np.concatenate((z1, z2))
    colors = colors1 + colors2
    plt.cla()
    plot_bars(x, y, z, thickness, colors, alpha=alpha)
    plt.legend(proxies, names)
    return nw, se, dfboth, x, y, z, thickness, colors



if __name__ == '__main__' and G.xls_file not in ['False', 'None']:
    rectangles = migmin_rectangles.migmin_rectangles()# todo
    df, _ = make_df_from_slope_file(G.slope_file,
                                    rectangles,
                                    find_plot.treatments,
                                    remove_redoings_time=3600)
    xlswrite_from_df(G.xls_file, df, True)



#
# def get_all_days_we_measured(resdict, minimum_number_of_measurements_in_a_day=1):
#     """ returns a list of (daynumber, number_of_measurements) pairs"""
#     from collections import Counter
#     daynr = [int(x['t']/86400) for x in res_list(resdict)]
#     counts = dict(Counter(daynr))
#     y = [(key, val) for (key, val) in counts.iteritems() if val>=minimum_number_of_measurements_in_a_day]
#     y.sort()
#     return y
#



# def excel_date(date1):
#     #from stackfoverflow
#     import datetime as dt
#     temp = dt.datetime(1899, 12, 30)
#     delta = date1 - temp
#     return float(delta.days) + (float(delta.seconds) / 86400)

# def excel_date2(date1):
#     delta = date1 - time.mktime((1899, 12,30))
#     return delta/86400
# def simplify_res_old(v):
#     v = defaultdict(lambda: None, v)
#     return {'name': v['name'],
#             'plot_nr': v['plot_nr'],
#             'heading': v['vehicle_pos']['heading'],
#             'vx': v['vehicle_pos']['x'],
#             'vy': v['vehicle_pos']['y'],
#             'vz': v['vehicle_pos']['z'],
#             'vsides': v['vehicle_pos']['side'],
#             't': v['t'],
#             'cx': v['chamber_pos']['x'],
#             'cy': v['chamber_pos']['y'],
#             'date': v['date'].replace('-', ''),
#             'slope_n2o': v['slopes']['N2O'],
#             'slope_co2': v['slopes']['CO2'],
#             'side': v['side']}


# def reslist2pandas_old(reslist):
#     lst = [simplify_res(x) for x in reslist]
#     return pd.DataFrame(lst)


