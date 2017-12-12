"""

Showing some examples of how to deal with the ffr data

"""

# First, select this file's directory in the white bar up to the
# right in Spyder

# You can step through this file by clicking on the arrow-buttons
# above. The one that runs current "cell" executes the code highlighted yellow
# (Cells are code between lines starting with # %%)


# %% Imports:
import os
import time
import glob
import numpy as np
import pandas as pd
import plotting_compat
from plotting_compat import plt
if hasattr(plotting_compat, 'get_ipython'):
    from plotting_compat import get_ipython
import resdir
import get_data
import utils
import regression
import find_regressions
from polygon_utils import plot_rectangles
import sort_results as sr
import divide_left_and_right
import weather_data
import flux_calculations
import scipy.stats
from statsmodels.formula.api import ols, rlm
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from scipy.stats import norm
import pH_data
import bucket_depths

def cla():
    if not plt.get_backend().endswith('inline'):
        plt.cla()

def clf():
    if not plt.get_backend().endswith('inline'):
        plt.clf()
        
# %%

# EDIT THESE:

# Select which rectangles, treatments and files you want:

# import migmin
# rectangles = migmin.migmin_rectangles()
# treatments = migmin.treatments
# data_file_filter_function = migmin.data_files_rough_filter

# or

# rectangles = something.agropro_rectangles()

# or

import buckets
rectangles = buckets.functions
treatments = buckets.treatments
data_file_filter_function = buckets.data_files_rough_filter

# Override the default result directories:
# (remember double backslashes)
resdir.raw_data_path = 'c:\\zip\\sort_results\\results'
slope_filename = 'c:\\zip\\sort_results\\bucket_slopes.txt'
# resdir.slopes_path = 'c:/users/larsmo/downloads'

# How to do regressions: This makes the "regressor object" regr which will be
# used further below.  It contains the functions and parameters for doing the
# regressions.  The parameters are collected in the dict named options;
# organizing the code this way makes it easier to replace the regression
# function with your own functions. Here, the options have:
# 'interval': the length of the regression interval (seconds)
# 'crit': how to choose the best interval within the run (for example the best
# 100 seconds within 180 seconds). 'crit' can be 'steepest' or 'mse' (mean squared error)
# 'co2_guides': wether or not the N2O regression will be done on the same interval
# as the CO2 regression.

options = {'interval': 100, 'crit': 'steepest', 'co2_guides': True}
regr = find_regressions.Regressor(slope_filename, options)

# regressions may take a long time. Set redo_regressions to False if you want to
# just use the slope file without redoing regression
redo_regressions = True

# END EDIT THESE


example_file = '2016-06-16-10-19-50-x599234_725955-y6615158_31496-z0_0-h0_743558650162_both_Plot_9_'


# Plotting the rectangles (not for buckets)
cla()
plot_rectangles(rectangles)
# with treatments:
cla()
keys = list(rectangles)
r = [rectangles[k] for k in keys]
tr = [treatments[k] for k in keys]
plot_rectangles(r, tr)


# %% some plotting examples
x = [1, 2, 3, 4]
y = [1, 3, 2, 4]
plt.plot(x, y)
plt.show()
plt.axis('equal')
plt.axis('auto')
s1 = plt.subplot(2, 3, 1)
x = np.linspace(0, 3 * np.pi)
plt.plot(x, np.sin(x))
plt.show()

# %%
clf()
plt.subplot(1, 1, 1)


# %% Get a list of all result files
filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
print("number of files: %d" % len(filenames))


# Get the data from file number 1000
a = get_data.get_file_data(filenames[1000])
cla()
plt.plot(a['N2O'][0], a['N2O'][1], '.')
plt.show()

# example_file has some fluxes:
filename = os.path.join(resdir.raw_data_path, example_file)
# checking that it exists first to avoid that this script stops:
if os.path.isfile(filename):
    a = get_data.get_file_data(filename)
    cla()
    plt.plot(a['N2O'][0], a['N2O'][1], '.')
    plt.show()
else:
    print('skipping example file')
    print('(does %s exist?)' % filename)


# %% Do a regression:
cla()
data = get_data.get_file_data(filename)
reg = regr.find_all_slopes(data, plotfun=plt.plot)

find_regressions.print_reg(reg)
plt.show()
# (we may also say reg = regr.find_all_slope(filename))
# Interval is the length of time of the regression line. crit can be 'steepest'
# or 'mse'; regressions will be done where the curves are steepest or where
# they have the lowest mse, respectively. If co2_guides==True, the interval in
# time where the co2 curve is the steepest or has the best mse is used for the
# time of regression for the N2O.


# (the division of the data from the two chambers is done in divide_left_and_right.py like this:
# import divide_left_and_right
# ad = divide_left_and_right.group_all(a)
# )

# %% Do many regressions

all_filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
filenames = data_file_filter_function(all_filenames)
print('number of raw data files:', len(filenames))
# this may take a long time:
if redo_regressions:
    regr.find_regressions(filenames)
else:
    # update resfile without redoing regressions:
    regr.update_regressions_file(filenames)


# %% Sort results according to the rectangles, put them in a Pandas dataframe
pd.set_option('display.width', 120)
# The slopes have been stored in the file whose name equals the value of
# slope_filename. make_df_from_slope_file picks the ones that are inside
# rectangles


df, df0 = sr.make_df_from_slope_file(slope_filename,
                                     rectangles,
                                     treatments,
                                     remove_redoings_time=3600,
                                     remove_data_outside_rectangles=True)

# %% Pandas... Pandas is a python library that gives python R-like
# dataframes. It takes some time to learn Pandas, although there is an
# introduction called "10 minutes to Pandas"
print(df.head())
print(df.tail())
print(df.columns)
print(df.date.min())
print(df.date.max())
print(df[['t', 'x', 'y', 'plot_nr']].head(10))
print(df[df.treatment == 'Control']
      [['t', 'x', 'y', 'plot_nr', 'treatment']].head(10))
pnr = df.plot_nr.values[0]
print('plotting N2O slopes for plot_nr', pnr)
d = df[df.plot_nr == pnr]
cla()
plt.axis('auto')
plt.plot(d['t'], d['N2O_slope'],'.-')
plt.show()
print(d['N2O_slope'])

# %% finally add in some weather data, calculate fluxes, wrap it up in
# a function. (if you have internet, you can do
# weather_data.data.update() first. This will download weather data
# from yr and save them)

weather_data.data.update()


def finalize_df(df, precip_dt=2):
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    # df['precip2'] = weather_data.data.get_precip2(df.t, [0, precip_dt]) #todo failed
    N2O_N_mol_secm2 = flux_calculations.calc_flux(df.N2O_slope, df.Tc)
    df['N2O_N_mmol_m2day'] = 2 * 1000 * 86400 * \
        N2O_N_mol_secm2  # 2 because 2 N in N2O
    CO2_C_mol_secm2 = flux_calculations.calc_flux(df.CO2_slope, df.Tc)
    df['CO2_C_mmol_m2day'] = 1000 * 86400 * CO2_C_mol_secm2
    df = sr.rearrange_df(df)
    return df


df = finalize_df(df)

print(df.head())


# %% A little check that the sorting is ok:
# Look at ginput-examples below to see how we can click on the dots to see
# where the outliers (if any) are from
def test_nrs(df, plot_numbers):
    plot_rectangles(rectangles, names=True)
    for nr in plot_numbers:
        d = df[df.plot_nr == nr]
        plt.plot(d.x, d.y, '.')


cla()

test_nrs(df, sorted(set(df.plot_nr)))
plt.show()

# %% Just the days with high fluxes:

df2 = sr.filter_for_average_slope_days(df, 0.0005)


# %% Excel.
# "..\filename.xls" makes filename.xls in the parent directory (..\)

openthefineapp = False
df.to_excel('..\excel_filename.xls')
if openthefineapp:
    os.system('..\excel_filename.xls')
# or
towrite = ['N2O_slope', 'CO2_slope', 'name']
sr.xlswrite_from_df('..\excel_filename2.xls', df, openthefineapp, towrite)
# or if you want it all:
sr.xlswrite_from_df('..\excel_filename3.xls', df, openthefineapp, df.columns)

# todo more sheets, names, small rectangles?

# useful:

a = df.groupby('daynr').N2O_slope
a = a.mean().values[a.count().values > 10]
cla()
plt.hist(a * 1000, bins='auto')
plt.show()
# %%barmaps

# _ = sr.barmap_splitted(df, theta=0)

# %% trapezoidal integration to calculate the emitted N2O over a period of time:


def trapz_df(df, column='N2O_N_mmol_m2day', factor=1 / 86400):
    index = sorted(set(df.plot_nr))
    trapzvals = []
    treatments = []
    for nr in index:
        d = df[df.plot_nr == nr]
        trapzvals.append(np.trapz(d[column], d.t) * factor)
        treatments.append(d.treatment.values[0])
    return pd.DataFrame(index=index,
                        data={'trapz': trapzvals,
                              'treatment': treatments,
                                  'plot_nr': index})


print(trapz_df(df))


# for buckets we want to test for effect of side:
def trapz_buckets(df, column='N2O_N_mmol_m2day', factor=1 / 86400):
    dleft = trapz_df(df[df.side == 'left'], column, factor)
    dleft['side'] = 'left'
    dright = trapz_df(df[df.side == 'right'], column, factor)
    dright['side'] = 'right'
    return pd.concat([dleft, dright])


# ordinary least squares regression on the trapz
# C(treatment) means that treatment is a categorical variable.
# Doing a log transform with a little addition to improve the normality tests
# %%
print('\nWith side as a factor (suitable for buckets):')
df_trapz = trapz_buckets(df)
model = 'np.log(trapz + 0.005) ~ C(treatment) + C(side)'
ols_trapz_res = ols(model, data=df_trapz).fit()
print(ols_trapz_res.summary())
# %%
print('\nWithout side as a factor')
df_trapz = trapz_df(df)
model = 'np.log(trapz + 0.005) ~ C(treatment)'
ols_trapz_res = ols(model, data=df_trapz).fit()
print(ols_trapz_res.summary())
# %%


def test(startdate='20170000', stopdate='2020'):
    df2 = df[(startdate <= df.date) & (df.date < stopdate)]
    df_trapz = trapz_buckets(df2)
    model = 'np.log(trapz + 0.005) ~ C(treatment) + C(side)'
    ols_trapz_res = ols(model, data=df_trapz).fit()
    print(ols_trapz_res.summary())
    print(len(df2))


# %% plotting buckets or migmin: todo move to another file

def plotnr(df, nr, t0):
    """ plotting bucket number nr in df, subtracting t0 from the time axis"""
    l = df[df.side == 'left'][df.plot_nr == nr]
    r = df[df.side == 'right'][df.plot_nr == nr]
    # l = left[df.plot_nr==i]
    # r = right[df.plot_nr==i]
    t = (l.t - t0) / 86400
    plt.plot(t, l['N2O_N_mmol_m2day'], '.-', t, r['N2O_N_mmol_m2day'], 'r.-')


def set_ylims(lims, nrows=6, mcols=4):
    """changing all the y axis limits to limits. Example set_ylims([0, 0.05])"""
    for i in range(nrows * mcols):
        plt.subplot(nrows, mcols, i + 1)
        plt.gca().set_ylim(lims)


def plot_treatment(df, treatment, row, t0,
                   delete_xticks=False, title=False):
    """ plotting all plots with given treatment in a row of subplots,
    subtracting t0 from the time axis.
    row is a list, e.g. [3,6], meaning row 3 of 6"""

    nr = sorted(set(df[df.treatment == treatment].plot_nr))
    for i, n in enumerate(nr):
        plt.subplot(row[1], len(nr), i + 1 + (row[0] - 1) * len(nr))
        plotnr(df, n, t0)
        plt.grid(True)
        if delete_xticks:
            plt.gca().set_xticklabels([])
        else:
            plt.gca().set_xlabel('day of year')
        if title:
            plt.gca().set_title('replicate %d' % (i + 1))


def plot_all(df, ylims=True, t0=(2017, 1, 1, 0, 0, 0, 0, 0, 0)):
    if isinstance(t0, (list, tuple)):
        t0 = time.mktime(t0)
    clf()
    tr = sorted(set(df.treatment))
    nrows = len(tr)
    for i, t in enumerate(tr):
        plot_treatment(df, t, [i + 1, nrows], t0, i < len(tr) - 1, i == 0)
    if ylims:
        set_ylims([df.N2O_N_mmol_m2day.min(), df.N2O_N_mmol_m2day.max()])
    for i, t in enumerate(tr):
        plt.subplot(6, 4, i * 4 + 1)
        plt.gca().set_ylabel(t)
        # mp.plot('text', (min(df.t)-t0)/86400, 0.1, t)


try:
    plot_all(df)
    plt.show()
except:
    pass

# %% barplots:


def barplot_trapz(df, sort_by_side=False):
    if sort_by_side:
        a = trapz_buckets(df)
    else:
        a = trapz_df(df)
    cla()
    # df.sort_index()
    treatments = sorted(a.treatment.unique())
    toplotx = []
    toploty = []
    toplot_colors = []
    ticx = []
    x = 1
    for i, tr in enumerate(treatments):
        b = a[a.treatment == tr]
        if sort_by_side:
            left = b[b.side == 'left'].trapz.values
            right = b[b.side == 'right'].trapz.values
            toplotx.extend(list(range(x, x + len(left) + len(right))))
            ticx.append(x + 2)
            x += 2 + len(left) + len(right)
            toploty.extend(list(left) + list(right))
            toplot_colors.extend(['r'] * len(left) + ['b'] * len(right))
        else:
            both = b.trapz.values
            toplotx.extend(list(range(x, x + len(both))))
            ticx.append(x + 2)
            x += 2 + len(both)
            toploty.extend(list(both))
            toplot_colors = 'b'
    plt.bar(toplotx, toploty, color=toplot_colors)
    plt.gca().set_xticks(ticx)
    plt.gca().set_xticklabels(treatments, rotation=30)
    plt.grid(True)
    plt.gca().set_ylabel('$\mathrm{g/m^2}$  maybe')
    return toplotx, toploty


clf()
plt.subplot()
a, b = barplot_trapz(df, True)
plt.show()

# %% Plot pH vs flux

ph_df = pH_data.df
#ph_df.groupby('nr').last()


def add_get_ph(df, ph_df, ph_method='CaCl2'):
    ph_df['plot_nr'] = ph_df['nr']
    tralala = ph_df.groupby('nr').last()
    def get_ph(nr):
        return tralala[tralala.plot_nr == nr][ph_method].values[0]
    df['pH'] = df.plot_nr.apply(get_ph)


def plot_ph_vs_flux(df_trapz, ph_df, ph_method='CaCl2'):
    a = df_trapz
    add_get_ph(a, ph_df)
    tr = sorted(a.treatment.unique())
    toplot = []
    markers = '*o><^.'
    for i, t in enumerate(tr):
        toplot.append(list(a.pH[a.treatment == t].values))
        toplot.append(list(a.trapz[a.treatment == t].values))
        toplot.append(markers[i])
    plt.plot(*toplot, markersize=8)
    plt.legend(tr)
    plt.gca().set_xlabel('pH (measurement in field %s)' % ph_df.date.max())
    plt.gca().set_ylabel('$\int_{t_0}^{t_1} \mathrm{flux\  dt}$')
    plt.grid(True)

cla()
plot_ph_vs_flux(trapz_df(df), ph_df)
plt.show()'

# %% subplots integration gothrough

# %% ginput-examples
# For this, you first need to enter
# % matplotlib auto
# or enter the command
# get_ipython().magic('matplotlib auto')
# in spyder. This makes the plot come up in a separate window, where they can
# be clicked on. (The plot window sometimes shows up behind the spyder window.)
# Enter "%matplotlib inline" when you want the inline plots back.
# We had

get_ipython().magic('matplotlib auto')
time.sleep(3)

def test_nrs(df, plot_numbers):
    plot_rectangles(rectangles, names=True)
    for nr in plot_numbers:
        d = df[df.plot_nr == nr]
        plt.plot(d.x, d.y, '.')

# We want to click on the dots and see the data associated with them


def ginput_show_info(df, fun=None, x='x', y='y'):
    print('Locate the plot window, click on a dot, or double-click (or triple-click) to quit')
    double_click_time = 0.5
    minimum_distance_index = None
    t0 = time.time()
    while 1:
        xy = plt.ginput(1)
        xy = xy[0]
        previous_one = minimum_distance_index
        if xy[0] is None or time.time() - t0 < double_click_time:
            break
        distances = np.sqrt((df[x] - xy[0])**2 + (df[y] - xy[1])**2)
        minimum_distance_index = distances.argmin()
        print(df.loc[minimum_distance_index])
        if fun:
            fun(df.loc[minimum_distance_index])
        t0 = time.time()
    return df.loc[previous_one] if previous_one else None


def show_reg_fun(row):
    plt.subplot(2, 1, 2)
    cla()
    filename = os.path.join(resdir.raw_data_path, row['name'])
    data = get_data.get_file_data(filename)
    regr.find_all_slopes(data, plotfun=plt.plot)


def test2(df):
    clf()
    plt.subplot(2, 1, 1)
    test_nrs(df, sorted(set(df.plot_nr)))
    return ginput_show_info(df, show_reg_fun)


test2(df)

# kind of the same, but plotting the slopes in the upper subplot


def test2(df):
    clf()
    plt.subplot(2, 1, 1)
    plt.plot(df.t, df.N2O_slope)
    return ginput_show_info(df, show_reg_fun, x='t', y='N2O_slope')


# %% making your own regression function:

# So you want to make your own regression function. We can do this like so (inheriting from the Regressor class):


class MyRegressor(find_regressions.Regressor):

    def find_all_slopes(self, filename_or_data, plotfun=None):
        """Returns a dict of regressoion objects.

        A regression object has slots named intercept, slope, se_intercept,
        se_slope, mse, start, and stop Each of these is a float. se is
        standard error, mse is mean squared error.
        """
        reg = regression.regression2
        # regression.regression2 is a function which returns a regression object
        if isinstance(filename_or_data, str):
            data = get_data.get_file_data(filename_or_data)
        else:
            data = filename_or_data
        # data is a dict with keys 'CO2', 'N2O', 'Wind', and so on
        # data['CO2'][0] is a list of seconds
        # data['CO2'][1] is a list of ppmv values
        # similarly for 'N2O'
        # Wind is a little bit different; it has not ppmv, but m/s
        # We show here just the simple linear regression using all the data,
        # but wind and self.options is available and may be used in this function,
        print('this is my own regression function')
        print('self.options is:', self.options)
        print('wind was on average:', np.mean(data['Wind'][1]), 'm/s')
        resdict = divide_left_and_right.group_all(data)
        co2 = resdict['CO2']
        n2o = resdict['N2O']
        regressions = dict
        regressions['left'] = dict
        regressions['right'] = dict
        regressions['left']['CO2'] = reg(co2['left'][0], co2['left'][1])
        regressions['left']['N2O'] = reg(n2o['left'][0], n2o['left'][1])
        regressions['right']['CO2'] = reg(co2['right'][0], co2['right'][1])
        regressions['right']['N2O'] = reg(n2o['right'][0], n2o['right'][1])
        return regressions


regr2 = MyRegressor('another_slopes_filename', {'p1': 100, 'p2': 3.14})
data = get_data.get_file_data(os.path.join(resdir.raw_data_path, example_file))
reg = regr2.find_all_slopes(data, plotfun=plt.plot)


# * batch-programmer
# I will at least temporarily remove some of these to make the code more tidy
# From dos, powershell or bash:

# >> cd to_the_path_to_this_readme_file

# >> python ffr_trace_plotter.py

# (you can untick the plot tickbox for faster regressions)

# Some command-line scripts. (-h gives a short help text)

# >> python find_regressions.py -h

# >> python sort_results.py -h

# >> python export_raw_data_to_excel.py -h

# For example

# >> python find_regressions.py c:\data --out c:\regressions\slopes.txt

# >> python sort_results.py  -s ..\slopes.txt
