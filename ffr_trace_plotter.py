#!/usr/bin/env python
import math
import sys
import os
import time
import copy
import itertools
import matplotlib
from collections import OrderedDict, namedtuple
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import LEFT, RIGHT, CENTER, BOTH, TOP, BOTTOM, YES, NO
import licor_indexes
import dlt_indexes
import findfile
import get_data
import app_open
import pickle
import traceback
import argparse
import find_regressions

defaultfile = ['d:/temp/New_folder/results/results/2015-06-15-14-38-05-x599226_260722-y6615166_83654-z0_0_right_Plot_9_']
resfile = 'slopes.txt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting and finding slopes')
    parser.add_argument('-f', '--file', type=str, default=defaultfile[0])
    parser.add_argument('--resfile', type=str, default=resfile,
                        help='Default: slopes.txt')
    args = parser.parse_args()
    defaultfile[0] = args.file


def timesleep(t):
    try:
        time.sleep(t)
    except KeyboardInterrupt:
        exit()


def make_table(res, unit):
    # print make_table({'left':{'N2O':1, 'CO2':2}, 'right':{'CO2':3, 'CO':4}})
    def getslope(side, name):
        if side in res and name in res[side]:
            return flux_estimate(name, res[side][name].slope, unit)
        else:
            return float('nan')
    substances = ['N2O', 'CO2', 'CO']
    n = max([len(x) for x in substances])
    m = 10
    sides = ['left', 'right']
    s = '{:{n}s} {:>{m}s} {:>{m}s}'.format('', 'left', 'right', n=n, m=m)
    for name in substances:
        left = getslope('left', name)
        right = getslope('right', name)
        s += '\n{:{n}s}{:{m}.3g} {:>{m}.3g}'.format(
            name, left, right, n=n + 2, m=m)
    return s


def flux_estimate(key, b1, unit):
    # b1 is ppm/sec
    chamber_height = 0.5  # meters
    molvol = 24e-3  # m3/mol
    molwt = {'N2O': 2 * 14, 'CO2': 12, 'CO': 12}
    flux = chamber_height * b1 * 1e-6 / molvol  # mol/m2/sec
    if unit == 'kg/ha/yr':
        return flux * 86400 * 365.35 * molwt[key] * 10000 / 1000
    else:
        return flux * 86400 * 1e6


# making one big class

class App(object):

    def __init__(self, master, files):
        self.master = master
        self.thefiles = files
        self.resfile = args.resfile

        self.filegui_frame = tk.Frame(master)
        self.filegui_frame.pack(fill=BOTH, side=BOTTOM, expand=YES)

        self.filename_frame = tk.Frame(master)
        self.filename_frame.pack(fill=BOTH, side=BOTTOM, expand=YES)

        self.axis_frame = tk.Frame(master)
        self.axis_frame.pack(side=LEFT, fill=BOTH, expand=YES)

        self.gui_frame = tk.Frame(master)
        self.gui_frame.pack(fill=BOTH, side=RIGHT, expand=NO)

        self.button_frame = tk.Frame(self.gui_frame)
        self.button_frame.pack()

        self.name_frame_frame = tk.Frame(self.gui_frame)
        self.name_frame_frame.pack(fill=BOTH, expand=YES)

        # Make the plot axis with toolbar
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.plot_axis = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.axis_frame)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=1)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.axis_frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)

        self.name_frame = [None]
        self.options = OrderedDict()
        self.retrieved_data = None
        self.regression_plots = {'N2O': [], 'CO2': [], 'CO': []}
        self.regression_values = {'N2O': 0, 'CO2': 0, 'CO': 0}
        self.regression_signature = {'N2O': 0, 'CO2': 0, 'CO': 0}
        self.flux_unit_list = ['\u00B5mol/m2/day', 'kg/ha/yr']

        self.filename_str = tk.StringVar()
        self.message_string = tk.StringVar()
        self.flux_string = tk.StringVar()
        self.unit_string = tk.StringVar()
        self.regression_time_string = tk.StringVar()
        self.lag_time_string = tk.StringVar()

        self.do_plot = tk.IntVar()
        self.scale = tk.IntVar()
        self.scale_more = tk.IntVar()
        self.normalize = tk.IntVar()
        self.show_legend = tk.IntVar()
        self.use_co2_as_guide = tk.IntVar()
        self.do_estimate_flux = tk.IntVar()
        self.running_regressions = False
        self.rowi = 0
        self.grid(tk.Checkbutton, text='Plot', variable=self.do_plot)
        self.grid(tk.Checkbutton, text='Scale', variable=self.scale,
                  command=self.make_fun(self.scale))
        self.grid(tk.Checkbutton, text='Finer', variable=self.scale_more,
                  command=self.make_fun(self.scale_more))
        self.grid(tk.Checkbutton, text='Normalize', variable=self.normalize,
                  command=self.make_fun(self.normalize))
        self.grid(tk.Checkbutton, text='Legend',
                  variable=self.show_legend, command=self.do_the_plotting)
        self.grid(tk.Checkbutton, text='CO2 decides\nN2O regression',
                  variable=self.use_co2_as_guide)
        self.grid(tk.Label, text='N2O-lag:')
        self.grid(tk.Entry, textvariable=self.lag_time_string, width=10,
                  insertofftime=0, justify=CENTER)
        self.grid(tk.Label, text='Regression-time:')
        self.grid(tk.Entry, textvariable=self.regression_time_string, width=10,
                  insertofftime=0, justify=CENTER)
        self.grid(tk.Button, text='Estimate flux',
                  command=self.find_fluxes, width=10)
        widget = tk.Checkbutton(
            self.button_frame, variable=self.do_estimate_flux)
        widget.grid(row=self.rowi - 1, column=1, stick='e')
        self.grid(tk.Label, textvariable=self.message_string)
        self.grid(tk.Button, text='Change units',
                  command=self.rotate_flux_unit_list)
        temp = tk.Label(self.button_frame, textvariable=self.unit_string)
        temp.grid(row=self.rowi - 1, column=1, stick='w')
        temp = tk.Label(self.button_frame, textvariable=self.flux_string,
                        justify=tk.LEFT, font=('Courier New', 8))
        temp.grid(row=self.rowi, column=0, columnspan=2)
        self.rowi += 1
        temp = tk.Label(self.filename_frame, textvariable=self.filename_str)
        temp.grid(row=0, column=0, stick='w')
        #grid2(tk.Label,  3,0, textvariable = filename_str)
        self.grid2(tk.Button, 1,0, text='File', command=self.readfile, width = 5)
        self.grid2(tk.Button, 1,1, text='<<', command=self.first_file, width = 5)
        self.grid2(tk.Button, 1,2, text='<', command=self.previous_file, width = 5)
        self.grid2(tk.Button, 1,3, text='>', command=self.next_file, width = 5)
        self.grid2(tk.Button, 1,4, text='>>', command=self.last_file, width = 5)
        self.grid2(tk.Button, 1,5, text='Run', command=self.run_regression, width = 5)
        self.grid2(tk.Button, 1,6, text='Stop', command=self.stop_running_regressions, width = 5)
        self.grid2(tk.Button, 1,7, text='Resfile', command=self.tidy_result_file, width = 8)
        self.show_legend.set(1)
        self.regression_time_string.set('50')
        self.lag_time_string.set('0')
        self.unit_string.set(self.flux_unit_list[0])
        self.normalize.set(True)
        self.do_plot.set(True)
        self.use_co2_as_guide.set(True)
        self.do_estimate_flux.set(True)
        self.master.after(1000, self.readfile)
        self.master.mainloop()

    def make_name_frame(self):
        try:
            self.name_frame.destroy()
        except:
            pass
        self.name_frame = tk.Frame(self.name_frame_frame)
        self.name_frame.pack(fill=BOTH, expand=YES)

    def make_name_fields(self, names):
        if names == list(self.options.keys()):
            return
        self.options.clear()
        self.make_name_frame()
        nf = self.name_frame
        tk.Label(nf, text='Name').grid(row=0, column=0)
        tk.Label(nf, text='plot').grid(row=0, column=1)
        tk.Label(nf, text='base').grid(row=0, column=2)
        for j, name in enumerate(names):
            i = j + 1
            var0 = tk.StringVar()
            var0.set(name)
            tk.Label(nf, textvariable=var0).grid(row=i, column=0)
            var1 = tk.IntVar()
            var1.set(1)
            cbut = tk.Checkbutton(
                nf, variable=var1, command=self.do_the_plotting)
            cbut.grid(row=i, column=1)
            var2 = tk.IntVar()
            var2.set(0)
            cbut = tk.Checkbutton(
                nf, variable=var2, command=self.do_the_plotting)
            cbut.grid(row=i, column=2)
            self.options[name] = {'name': var0,
                                  'plot': var1, 'subtract_minimum': var2}
        tk.Label(nf, text='All on/off').grid(row=i + 1, column=0)
        b = tk.Button(nf, text=' ^ ',
                      command=lambda: self.all_on_or_off('plot'))
        b.grid(row=i + 1, column=1)
        b = tk.Button(nf, text=' ^ ',
                      command=lambda: self.all_on_or_off('subtract_minimum'))
        b.grid(row=i + 1, column=2)

    def all_on_or_off(self, s):
        all_on = all([self.options[key][s].get() for key in self.options])
        for _, opt in self.options.items():
            opt[s].set(1 if not all_on else 0)
        self.do_the_plotting()

    def normalized(self, x):
        # x is list of lists
        if len(x) == 0:
            return x
        x0 = min([min(xi) for xi in x])
        x1 = max([max(xi) for xi in x])
        if x0 == x1:
            return x
        else:
            return [[(a - x0) * 1.0 / (x1 - x0) for a in xi] for xi in x]

    def find_scaling_divisor(self, y):
        if len(y) == 0:
            return 1
        temp = max([abs(x) for x in y])
        if temp == 0:
            divisor = 1
        else:
            maxabsy = 1.0 * max([abs(x) for x in y])
            divisor = 10.0**(math.ceil(math.log10(maxabsy))) / 10.0
            if self.scale_more.get():
                if maxabsy / divisor < 2:
                    divisor /= 5
                elif maxabsy / divisor < 5:
                    divisor /= 2
        divisor_string = '' if divisor == 1 else ' /' + "%g" % divisor
        return divisor, divisor_string

    def scale_xydata(self, xydata):

        def scale_y(y, divisor):
            return [x * 1.0 / divisor for x in y]

        for key in self.options:
            yy = xydata[key][1::2]
            if self.scale.get() or self.scale_more.get():
                divisor, divisor_string \
                    = self.find_scaling_divisor(list(itertools.chain(*yy)))
                xydata[key][1::2] = [scale_y(y, divisor) for y in yy]
            else:
                divisor_string = ''
            self.options[key]['name'].set(key + divisor_string)
        return xydata

    def subtract_min(self, xydata):
        for key, opt in self.options.items():
            if opt['subtract_minimum'].get():
                mn = min([min(x) for x in xydata[key][1::2]])
                xydata[key][1::2] = [[y - mn for y in ylist]
                                     for ylist in xydata[key][1::2]]
        return xydata

    def lag_adjust_xydata(self, xydata):
        if 'N2O' in xydata:
            dt = self.get_lag_time()
            xydata['N2O'][::2] = [[t - dt for t in tlist]
                                  for tlist in xydata['N2O'][::2]]
        return xydata

    def add_on_regressions(self, xydata):
        for key, xy in self.regression_plots.items():
            if xy and key in xydata:
                xydata[key].extend(xy)
        return xydata

    def do_the_plotting(self):
        xydata = copy.deepcopy(self.retrieved_data)
        for key in ['aux', 'side']:
            if key in xydata:
                xydata.pop(key)
        try:
            xydata = self.add_on_regressions(xydata)
            xydata = self.lag_adjust_xydata(xydata)
            self.message_string.set('')
            self.make_name_fields(list(xydata.keys()))
            xydata = self.subtract_min(xydata)
            xydata = self.scale_xydata(xydata)
            self.plot_axis.cla()
            try:
                self.plot_axis.plot(*self.choose_plot(xydata))
            except Exception as e:
                self.message_string.set(traceback.format_exc())
            self.plot_axis.grid(1)
            matplotlib.rcParams.update({'font.size': 10})
            if self.show_legend.get():
                self.plot_axis.legend(self.legend_names(), loc=0)
            self.fig.canvas.draw()
        except Exception as e:
            self.message_string.set(traceback.format_exc())

    def choose_plot(self, xydata):
        xy = []
        regression_xy = []
        # putting the regression lines at the end of the returned list so
        # the color of the time series don't change when the regression
        # lines are plotted
        for key, opt in self.options.items():
            if opt['plot'].get():
                new = xydata[key]
                if self.normalize.get():
                    new[1::2] = self.normalized(new[1::2])
                xy.extend(new[:2])
                if len(new) > 2:
                    regression_xy.extend(new[2:] + ['k'])
        return xy + regression_xy

    def legend_names(self):
        return [key for key in self.options if self.options[key]['plot'].get()]

    def rotate_flux_unit_list(self):
        self.flux_unit_list = self.flux_unit_list[-1:] + \
            self.flux_unit_list[:-1]
        self.unit_string.set(self.flux_unit_list[0])
        self.find_fluxes(False)

    def get_regression_time(self):
        default_t = 30
        a = self.regression_time_string.get()
        try:
            t = float(a)
        except:
            t = -1
        if t <= 0:
            t = 30
            self.regression_time_string.set(repr(t))
        return t

    def get_lag_time(self):
        default_t = 0
        a = self.lag_time_string.get()
        try:
            t = float(a)
        except:
            t = -1
        if t <= 0:
            t = default_t
            self.lag_time_string.set(repr(t))
        return t

    def find_fluxes(self, do_plot=True):
        time_interval = self.get_regression_time()  # sec that the regression will cover
        ret = ''
        keys = list(self.regression_plots.keys())
        co2_guides = self.use_co2_as_guide.get() and keys.count('CO2')
        res = find_regressions.find_all_slopes(
            self.retrieved_data, time_interval, co2_guides)
        # todo sjekke self.retrieved_data ok? res ok?
        # res['left'] and res['right'] should be dict with keys
        # 'CO2' and 'N2O' (or other substance names), values
        # (Regression, (x, y))
        for side in ('left', 'right'):
            for key in res[side] if side in res else []:
                self.regression_plots[key] = []
        for side in ('left', 'right'):
            for key, reg in iter(res[side].items()) if side in res else []:
                t, y = self.retrieved_data[key][:2]
                t0, t1 = t[reg.start], t[reg.stop]
                y0, y1 = reg.intercept + reg.slope * t0, reg.intercept + reg.slope * t1
                self.regression_plots[key].extend([[t0, t1], [y0, y1]])
                #                self.regression_signature[key] = y[0]
                #                if key=='CO2':
                #     dt = self.get_lag_time()
                #     co2_t = [t0+dt,t1+dt]
                # s += self.make_flux_strings(key, b1)
                # ret += '{0}\t{1}\t'.format(key, b1)
        self.flux_string.set(make_table(res, self.flux_unit_list[0]))
        if do_plot:
            self.do_the_plotting()
        return ret.strip('\t')

    def _readfile(self, filename, extra_string=''):
        with open(filename, 'rb') as f:
            a = pickle.load(f)
        self.retrieved_data = get_data.parse_saved_data(
            get_data.old2new(a), filename)
        self.filename_str.set('hei')
        self.filename_str.set(filename + extra_string)
        if self.do_estimate_flux.get():
            s = self.find_fluxes(do_plot=self.do_plot.get())
        else:
            self.do_the_plotting()
        splitname = os.path.split(filename)
        with open(self.resfile, 'a') as f:
            f.write(splitname[1] + '\t' + s + '\n')

    def readfile(self):
        f = self.thefiles.findfile()
        self._readfile(f)

    def first_file(self):
        self._readfile(self.thefiles.first())

    def next_file(self):
        f = next(self.thefiles)
        extra_string = ' (%d/%d)' % (self.thefiles.index +
                                     1, self.thefiles.nfiles)
        self._readfile(f, extra_string)

    def previous_file(self):
        self._readfile(self.thefiles.previous())

    def last_file(self):
        self._readfile(self.thefiles.last())

    def run_regression():
        self.running_regressions = True
        self.thefiles.update()

        def callback():
            if self.thefiles.index <= self.thefiles.nfiles - 2 \
               and self.running_regressions:
                self.next_file()
                self.master.after(10, callback)  # master er root
            else:
                print('done')
                self.do_the_plotting()
        callback()

    def stop_running_regressions(self):  # henger
        self.running_regressions = False

    def tidy_result_file(self):
        sys.stdout.flush()
        if not self.resfile:
            print('no result file yet')
            return
        with open(self.resfile) as f:
            lines = f.read().splitlines()
        lines = [x for x in lines if x]
        lines.reverse()
        lines2 = []
        keys = []
        for line in lines:
            x = line.split()
            if len(x) and x[0] not in keys:
                keys.append(x[0])
                lines2.append(line)
        lines2.reverse()
        with open(self.resfile, 'w') as f:
            f.write('\n'.join(lines2))
        # print lines
        # print lines2
        app_open.open(self.resfile)

    def message(self, s, t=0):
        self.message_string.set(repr(s))
        time.sleep(t)

    def make_fun(self, var, others=None):
        if others is None:
            others = [self.scale, self.scale_more, self.normalize]

        def f():
            if var.get():
                # unticks the other two
                for v in others:
                    if var != v:
                        v.set(0)
            self.do_the_plotting()
        return f

    def grid(self, fun, **args):
        widget = fun(self.button_frame, **args)
        widget.grid(row=self.rowi, column=0, stick='w')
        self.rowi += 1
        return widget

    def grid2(self, fun, row, col, **args):
        widget = fun(self.filegui_frame, **args)
        widget.grid(row=row, column=col, stick='w')
        return widget


def selection_fun(x):
    return x.startswith('20') or x.startswith('21') or x.startswith('punkt')


thefiles = findfile.File_list('', '_tobor_recent.txt',
                              selection_fun=selection_fun)


if __name__ == "__main__":

    root = tk.Tk()

    app = App(root, thefiles)


# def parse_data(data):# not used, though. todo
#     xydata = OrderedDict()
#     if isinstance (data[0][0], str):
#         for d in data:
#             if len(d)>2:
#                 if (len(d)%2)==0:
#                     raise RuntimeError, "odd number of data to plot"
#                 xydata[d[0]] = list(d[1:])
#             else:
#                 xydata[d[0]] = [range(len(d[1])), d[1]]
#     else:
#         if len(data)==1:
#             data = [range(len(data[0])), data[0]]
#         if len(data)%2==1:
#             raise RuntimeError, "odd number of data to plot"
#         for i in range(len(data)/2):
#             xydata[repr(i)] = list(data[2*i:2*i+2])
#     return xydata
