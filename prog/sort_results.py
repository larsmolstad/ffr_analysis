"""
"""

import os
import xlwt
import tkinter.messagebox
import tkinter.filedialog
import datetime
import argparse
from math import sin, cos, pi

import last_directory
from migmin import migmin_rectangles
import find_plot
from get_data import parse_filename
import numpy as np
import pandas as pd
import resdir
from plotting_compat import plt
plot = plt.plot


class G:  # (for global)
    slope_file = os.path.join(resdir.slopes_path, 'slopes.txt')
    xls_file = 'results.xls'
    chamber_distance = 2  # meters, sideways distance from gps to chamber center
    chamber_fw_distance = 0.2  # forward distance from gps to chamber center


description = """
Sorts the slopes according to position and writes the results to excel
Examples
python sort_results.py
python sort_results.py -s slopes.txt
python sort_results.py -s ..\slopes.txt --out resultfile.xls
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-s', '--slope_file', type=str, default='',
                        help='If not given, a dialog box opens. \n')
    parser.add_argument('--out', type=str, default='results.xls',
                        help='Result file. Default results.xls')
    args = parser.parse_args()
    G.slope_file = args.slope_file
    G.xls_file = args.out


def tk_getfilename(remember_file, title="open"):
    lastdir = last_directory.remember(remember_file)
    file = tkinter.filedialog.askopenfilename(initialdir=os.path.split(lastdir.get())[0],
                                              title=title)
    lastdir.set(file)  # os.path.split(file)[0])
    return file


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
    # df['id'] = df.index
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
    return df[tokeep + tomove]  # df.drop(todrop, axis=1, errors='ignore')


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
    reslist = reslist[start:stop] if stop >= 0 else reslist[start:]
    if index_list is not None:
        reslist = [x[i] for i in index_list]
    for s in reslist:
        str2num_line(s)
    return reslist


def make_df_from_slope_file(name,
                            rectangles,
                            treatment_dict,
                            remove_redoings_time=3600,
                            remove_data_outside_rectangles=True):
    unsorted_res = get_result_list_from_slope_file(name, start=0)
    df0 = make_df(unsorted_res)
    df0['plot_nr'] = find_plot.find_plots(df0, rectangles)
    df0['treatment'] = df0.plot_nr.map(lambda x: treatment_dict[x]
                                       if x in treatment_dict else None)
    translations = {'N2O': 'N2O_slope', 'CO': 'CO_slope', 'CO2': 'CO2_slope'}
    df0.rename(columns=translations, inplace=True)
    df = df0
    if remove_data_outside_rectangles:
        df = df[df.plot_nr > 0]
    df = rearrange_df(df)
    if remove_redoings_time:
        df = remove_redoings(df, remove_redoings_time)
    return df, df0


def plot_plots(df, plots, symbol='.', markersize=3, ret=None):
    a = []
    for key in plots:
        r = df[df.plot_nr == key]
        a.append(r.t / 86400)
        a.append(r.N2O_slope)
        a.append(symbol)
    plot(*a, markersize=markersize)
    return a if ret is not None else None


def filter_for_average_slope_days(df, lower=0.0001, upper=np.inf):
    """ makes a new dataframe containing only the results from the days
    where the average slopes were between lower and upper"""
    means = df.groupby('daynr').N2O_slope.mean()
    days = means[(means < upper) & (means >= lower)].index
    return df[df.daynr.isin(days)]


def find_nonlast_redoings(df, nr, dt=3600):
    """Finds where the robot has measured in the same plot (and on the
    same side) after less than dt (in seconds).  Ususally, if dt is
    small, this is due to something going wrong, and the measurement
    has been restarted.

    """
    d = df[df.plot_nr == nr]
    # sides = df0.loc[d.index].side
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


# we want to have empty cells for missing measurements in the
# spreadsheet.  problem with that: If the same plot is measured twice
# in a day it might be because all plots are measured twice or because
# the robot was stopped and restarted. Anyway I will have to sort them
# by days and make a rule for what to keep and what to insert as empty.
# Must go through the measurements day by day instead of plot by plot


def xlswrite_from_df(name, df, do_open=False, columns=['N2O_slope', 'CO2_slope']):
    # daynrs = sorted(set(df.daynr))# but sometimes we measure twice per day
    workbook = xlwt.Workbook()
    date_format = xlwt.XFStyle()
    date_format.num_format_str = 'dd/mm/yyyy'
    treatments = sorted(set(df.treatment))
    for compound in columns:  # todo name (not in df so need df0) and x and y
        w = workbook.add_sheet(compound)
        i = 0
        for treatment in treatments:
            d = df[df.treatment == treatment]
            plots = sorted(set(d.plot_nr))
            for j, plot_nr in enumerate(plots):
                i += 2
                w.write(0, i - 1, treatment)
                w.write(1, i - 1, str(plot_nr))
                d2 = d[d.plot_nr == plot_nr]
                t = d2.t.values
                y = d2[compound].values
                for rownr, ti in enumerate(t):  # todo vectorize?
                    ti = datetime.datetime.utcfromtimestamp(ti)
                    w.write(rownr + 2, i - 1, ti, date_format)
                    try:
                        w.write(rownr + 2, i, y[rownr])
                    except:
                        # sometimes I get
                        # Exception: Unexpected data type <class 'numpy.int64'>
                        # ,so
                        try:
                            w.write(rownr + 2, i, float(y[rownr]))
                        except Exception as e:
                            print(e)
                            print(float(y[rownr]))
    try:
        workbook.save(name)
    except IOError:
        raise IOError("You must close the old xls file")
    if do_open:
        os.startfile(name)


if __name__ == '__main__' and G.xls_file not in ['False', 'None']:
    lastfile = '.sort_result_lastfile'
    if not G.slope_file:
        G.slope_file = tk_getfilename(lastfile, "Select slope file")
    rectangles = migmin_rectangles.migmin_rectangles()  # todo
    df, _ = make_df_from_slope_file(G.slope_file,
                                    rectangles,
                                    find_plot.treatments,
                                    remove_redoings_time=3600)
    xlswrite_from_df(G.xls_file, df, True)
