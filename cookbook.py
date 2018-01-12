"""

Showing some examples of how to deal with the ffr data

"""

# First, select this file's directory in the white bar up to the
# right in Spyder. (Or else get "No module named ... " message)
# You can do this by right-clicking on "cookbook.py" above and choosing #
# "Set console working directory"

# You can step through this file by clicking on the arrow-buttons
# above. The one that runs current "cell" executes the code highlighted yellow
# (Cells are code between lines starting with # %%)


# %% Imports:
import sys
import os
import time
import glob
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 
sys.path.append(os.path.join(os.getcwd(), 'prog'))
import plotting_compat
import pylab as plt
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
import ginput_show
# import scipy.stats
from statsmodels.formula.api import ols#, rlm
# from statsmodels.stats.anova import anova_lm
# import statsmodels.api as sm
# from scipy.stats import norm
import pH_data
# import bucket_depths
plt.rcParams['figure.figsize'] = (10, 6)
current_path = os.getcwd()

# %% #################  EDIT THESE PARAMETERS: ################################

# Select which rectangles, treatments and files you want:

# import migmin as experiment

# or

# import buckets as experiment

# or

import agropro as experiment

# Override the default result directories:
# (remember double backslashes)

slopes_filename = experiment.slopes_filename
exception_list_filename = '..\\..\\exceptions.xlsx'
# resdir.raw_data_path = 'Y:\\MINA\\Miljøvitenskap\\Jord\\FFR\\results'
resdir.raw_data_path = '..\\..\\results'
# How to do regressions: The next two lines makes the "regressor
# object" regr which will be used further below.  It contains the
# functions and parameters for doing the regressions.  The parameters
# are collected in the dict named options. (Organizing the code this
# way makes it easier to replace the regression function with your own
# functions.)  Here, the options have: 'interval': the length of the
# regression interval (seconds) 'crit': how to choose the best
# interval within the run (for example the best 100 seconds within 180
# seconds). 'crit' can be 'steepest' or 'mse' (mean squared error)
# 'co2_guides': wether or not the N2O regression will be done on the
# same interval as the CO2 regression. Se the excel ex_options file
# for more options

options = {'interval': 100, 'crit': 'steepest', 'co2_guides': True}

regr = find_regressions.Regressor(slopes_filename, options, exception_list_filename)

# regressions may take a long time. Set redo_regressions to False if you want to
# just reuse the slope file without redoing regression. Regressions will still be
# done if options have changed.
redo_regressions =  False
# Choose flux units
# factor is the conversion factor from mol/s/m2 to the given unit
# flux_units = {'N2O': {'name': 'N2O_N_mmol_m2day', 'factor': 2 * 1000 * 86400},
#              'CO2': {'name': 'CO2_C_mmol_m2day', 'factor': 1000 * 86400}}

flux_units = {'N2O': {'name': 'N2O_N_mug_m2h', 'factor': 2 * 14 * 1e6 * 3600},
               'CO2': {'name': 'CO2_C_mug_m2h', 'factor': 12 * 1e6 * 3600}}

start_and_stopdate = ['201701', '3000']

excel_filename_start = experiment.name
# %% ################### END EDIT THESE PARAMETERS ############################

slopes_filename = utils.ensure_absolute_path(slopes_filename)
rectangles = experiment.rectangles
treatments = experiment.treatments
data_file_filter_function = experiment.data_files_rough_filter

treatment_names = sr.find_treatment_names(treatments)
example_file = '2016-06-16-10-19-50-x599234_725955-y6615158_31496-z0_0-h0_743558650162_both_Plot_9_'


#we'll do this a lot:
def with_raw_dir(filename):
    return os.path.join(resdir.raw_data_path, filename)

# Plotting the rectangles (not for buckets)
plt.cla()
plot_rectangles(rectangles)
# with treatments:
plt.cla()
keys = list(rectangles)
r = [rectangles[k] for k in keys]
tr = ['_'.join(treatments[k].values()) for k in keys]# todo
plot_rectangles(r, tr)
plt.show()

# %% some plotting examples
# plt.cla()
# x = [1, 2, 3, 4]
# y = [1, 3, 2, 4]
# plt.plot(x, y)
# plt.show()
# plt.axis('equal')
# plt.axis('auto')
# s1 = plt.subplot(2, 3, 1)
# x = np.linspace(0, 3 * np.pi)
# plt.plot(x, np.sin(x))
# plt.show()

# %%
plt.clf()
plt.subplot(1, 1, 1)


# %% Get a list of all result files
filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
print("number of files: %d" % len(filenames))

print("\nSome examples:")

def plot_raw(filename, key='N2O'):
    "key may be 'N2O', 'CO2', 'Wind'"
    if not os.path.isfile(filename):
        filename = with_raw_dir(filename)
    a = get_data.get_file_data(filename)
    plt.plot(a[key][0], a[key][1], '.')
    plt.gca().set_xlabel('seconds')
    if key in ['N2O', 'CO2', 'CO']:
        plt.gca().set_ylabel('ppm')
    return a

# Get the data from file number 1000 (or the last one if there are
# less than 1000 files)
n = 1000 if len(filenames)>1000 else len(filenames)
plt.cla()
a = plot_raw(filenames[n])
plt.show()

    
# example_file has some fluxes:
filename = with_raw_dir(example_file)
# checking that it exists first to avoid that this script stops:
if os.path.isfile(filename):
    plt.cla()
    plot_raw(filename, 'N2O')
    plt.show()
else:
    print('skipping example file')
    print('(does %s exist?)' % filename)

# also,
# a = get_data.get_file_data(filename)
# plt.plot(a['N2O'][0], a['N2O'][1], '.')

# %% Do a regression:
plt.cla()
reg = regr.find_all_slopes(filename, do_plot=True)
plt.show()
find_regressions.print_reg(reg)
# another way:
# data = get_data.get_file_data(filename)
# reg = regr.find_all_slope(data)

# %% Do many regressions

all_filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
filenames = data_file_filter_function(all_filenames, *start_and_stopdate)
print('number of raw data files:', len(filenames))
# this may take a long time:
if redo_regressions:
    regr.find_regressions(filenames)
else:
    # update resfile without redoing regressions:
    regr.update_regressions_file(filenames)

def plot_error_number(n, key='N2O'):
    name, err = find_regressions.regression_errors[-1][n]
    print('--------- name was: %s\nerror was:\n%s\n----------'%(name,err))
    a = plot_raw(name)
    print('shifting:', a['side'])
    return name, a

# %% Sort results according to the rectangles, put them in a Pandas dataframe
pd.set_option('display.width', 120)
# The slopes have been stored in the file whose name equals the value of
# slope_filename. make_df_from_slope_file picks the ones that are inside
# rectangles

df, df0 = sr.make_df_from_slope_file(slopes_filename,
                                     rectangles,
                                     treatments,
                                     remove_redoings_time=3600,
                                     remove_data_outside_rectangles=True)

df0 = df0[(df0.date >= start_and_stopdate[0]) & (df0.date <= start_and_stopdate[1])]

df = df[(df.date >= start_and_stopdate[0]) & (df.date <= start_and_stopdate[1])]

df0.sort_values('date', inplace=True)#todo flytte
df.sort_values('date', inplace=True)

# %% Pandas... Pandas is a python library that gives python R-like
# dataframes. It takes some time to learn Pandas, although there is an
# introduction called "10 minutes to Pandas". Try read that -- you should 
# understand the difference between df.loc[n] and df.iloc[n]. I never use df.ix.
# print(df.head())
# print(df.tail())
# print(df.columns)
print(df.date.min())
print(df.date.max())
# print(df[['t', 'x', 'y', 'plot_nr']].head(10))
# print(df[df.treatment == 'Control'][['t', 'x', 'y', 'plot_nr', 'treatment']].head(10))
# pnr = df.plot_nr.values[0]
# print('plotting N2O slopes for plot_nr', pnr)
# d = df[df.plot_nr == pnr]
# plt.cla()
# plt.axis('auto')
# plt.plot(d['t'], d['N2O_slope'], '.-')
# plt.show()
# print(d['N2O_slope'].tail())

## %% finally add in some weather data, calculate fluxes, wrap it up in
# a function. (if you have internet, you can do
# weather_data.data.update() first. This will download weather data
# from yr and save them)

weather_data.data.update()


def finalize_df(df, precip_dt=2):
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    # df['precip2'] = weather_data.data.get_precip2(df.t, [0, precip_dt]) #todo failed
    N2O_mol_secm2 = flux_calculations.calc_flux(df.N2O_slope, df.Tc)
    CO2_C_mol_secm2 = flux_calculations.calc_flux(df.CO2_slope, df.Tc)
    if experiment.name == 'buckets':
        N2O_mol_secm2 = N2O_mol_secm2 * (50/23.5)**2 * 0.94
        CO2_C_mol_secm2 = CO2_C_mol_secm2 * (50/23.5)**2 *0.94
    df['N2O_mol_m2s'] = N2O_mol_secm2
    df['CO2_mol_m2s'] = CO2_C_mol_secm2
    Nunits = flux_units['N2O']
    Cunits = flux_units['CO2']
    df[Nunits['name']] = Nunits['factor'] *  N2O_mol_secm2
    df[Cunits['name']] = Cunits['factor'] *  CO2_C_mol_secm2
    df = sr.rearrange_df(df)
    return df

df = finalize_df(df)

# print(df.head())


# %% A little check that the sorting is ok:
# Look at ginput-examples below to see how we can click on the dots to see
# where the outliers (if any) are from

def test_nrs(df, plot_numbers):
    plot_rectangles(rectangles, names=True)
    for nr in plot_numbers:
        d = df[df.plot_nr == nr]
        plt.plot(d.x, d.y, '.')


plt.cla()

test_nrs(df, sorted(set(df.plot_nr)))
plt.show()

# also, for example,
# test_nrs(df[df.treatment=='Larvikite'], sorted(set(df.plot_nr)))

# %% Just the days with high fluxes:

df2 = sr.filter_for_average_slope_days(df, 0.0005)


# %% Excel.
# "..\filename.xls" makes filename.xls in the parent directory (..\)

openthefineapp = False
excel_filenames = ['..\\'+excel_filename_start + '_' + s + '.xls' 
                   for s in 'df slopes all_columns'.split()]
df.to_excel(excel_filenames[0])
if openthefineapp:
    os.system(excel_filenames[0])
# or
towrite = ['N2O_slope', 'CO2_slope', 'filename']
sr.xlswrite_from_df(excel_filenames[1], df, openthefineapp, towrite)
# or if you want it all:
sr.xlswrite_from_df(excel_filenames[2], df, openthefineapp, df.columns)
print('Excel files written to parent directory')
# todo more sheets, names, small rectangles?

# %%barmaps

# _ = sr.barmap_splitted(df, theta=0)

# %% trapezoidal integration to calculate the emitted N2O over a period of time:

q = [0]
def trapz_df(df, column='N2O_mol_m2s', factor=14*2):
    # factor = 14*2 gives grams N
    index = sorted(set(df.plot_nr))
    trapzvals = []
    treatments = {name:[] for name in treatment_names}
    for nr in index:
        d = df[df.plot_nr == nr]
        trapzvals.append(np.trapz(d[column], d.t) * factor)
        for name in treatment_names:
            treatments[name].append(d[name].values[0])
    data = {'trapz': trapzvals,
            'plot_nr': index}
    for name in treatment_names:
        data[name] = treatments[name]
    q[0] = data
    return pd.DataFrame(index=index, data = data)


# for buckets we want to test for effect of side:
def trapz_buckets(df, column='N2O_mol_m2s', factor=14*2):
    # factor = 14*2 gives grams N
    dleft = trapz_df(df[df.side == 'left'], column, factor)
    dleft['side'] = 'left'
    dright = trapz_df(df[df.side == 'right'], column, factor)
    dright['side'] = 'right'
    return pd.concat([dleft, dright])


# ordinary least squares regression on the trapz
# C(treatment) means that treatment is a categorical variable.
# Doing a log transform with a little addition to improve the normality tests
# %%
# print('\nWith side as a factor (was suitable for buckets):')
# df_trapz = trapz_buckets(df)
# model = 'np.log(trapz + 0.005) ~ C(treatment) + C(side)'
# ols_trapz_res = ols(model, data=df_trapz).fit()
# print(ols_trapz_res.summary())
# %%
print('\nWithout side as a factor')
df_trapz = trapz_df(df)
model = 'np.log(trapz + 0.005) ~ C(rock_type)'
if len(treatment_names)==3:
    model = 'np.log(trapz + 0.5) ~ C(rock_type) + C(fertilizer) + C(mixture)'
    #model = 'np.log(trapz + 0.5) ~ C(rock_type) + C(fertilizer) + C(mixture) + C(rock_type)*C(fertilizer) + C(rock_type)*C(mixture) + C(fertilizer)*C(mixture)'
ols_trapz_res = ols(model, data=df_trapz).fit()
print(ols_trapz_res.summary())

print(df_trapz)
print('(trapz is g/m2)')
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
    l = df[(df.side == 'left') & (df.plot_nr == nr)]
    r = df[(df.side == 'right') & (df.plot_nr == nr)]
    # l = left[df.plot_nr==i]
    # r = right[df.plot_nr==i]
    lt = (l.t - t0) / 86400
    rt = (r.t - t0) / 86400
    name = flux_units['N2O']['name']
    plt.plot(lt, l[name], '.-', rt, r[name], 'r.-')


def set_ylims(lims, nrows=6, mcols=4):
    """changing all the y axis limits to limits. Example set_ylims([0, 0.05])"""
    for i in range(nrows * mcols):
        plt.subplot(nrows, mcols, i + 1)
        plt.gca().set_ylim(lims)


def plot_treatment(df, rock_type, row, t0,
                   delete_xticks=False, title=False):
    """ plotting all plots with given rock_type in a row of subplots,
    subtracting t0 from the time axis.
    row is a list, e.g. [3,6], meaning row 3 of 6"""

    nr = sorted(set(df[df.rock_type == rock_type].plot_nr))
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
    units = flux_units['N2O']['name']
    if isinstance(t0, (list, tuple)):
        t0 = time.mktime(t0)
    plt.clf()
    tr = sorted(set(df.treatment))
    nrows = len(tr)
    for i, t in enumerate(tr):
        plot_treatment(df, t, [i + 1, nrows], t0, i < len(tr) - 1, i == 0)
    if ylims:
        name = units
        set_ylims([df[name].min(), df[name].max()])
    for i, t in enumerate(tr):
        plt.subplot(6, 4, i * 4 + 1)
        plt.gca().set_ylabel(t)
        # mp.plot('text', (min(df.t)-t0)/86400, 0.1, t)
    print('\n(%s from %s to %s)' % (name, df.date.min(), df.date.max()))
    

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
    plt.cla()
    # df.sort_index()
    rock_types = sorted(a.rock_type.unique())
    toplotx = []
    toploty = []
    toplot_colors = []
    ticx = []
    x = 1
    for i, tr in enumerate(rock_types):
        b = a[a.rock_type == tr]
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
    plt.gca().set_xticklabels(rock_types, rotation=30)
    plt.grid(True)
    plt.gca().set_ylabel('$\mathrm{g/m^2}$  maybe')
    return toplotx, toploty


plt.clf()
plt.subplot()
do_split = experiment.name == 'buckets'
a, b = barplot_trapz(df, do_split)
plt.show()
print("try a, b = barplot_trapz(df[df.date>'2016'], True)")
# %% Plot pH vs flux

ph_df = pH_data.df
# ph_df.groupby('nr').last()


def add_get_ph(df, ph_df, ph_method='CaCl2'):
    ph_df['plot_nr'] = ph_df['nr']
    tralala = ph_df.groupby('nr').last()

    def get_ph(nr):
        return tralala[tralala.plot_nr == nr][ph_method].values[0]
    df['pH'] = df.plot_nr.apply(get_ph)


def plot_ph_vs_flux(df_trapz, ph_df, ph_method='CaCl2'):
    a = df_trapz
    add_get_ph(a, ph_df)
    tr = sorted(a.rock_type.unique())
    toplot = []
    markers = '*o><^.'
    for i, t in enumerate(tr):
        toplot.append(list(a.pH[a.rock_type == t].values))
        toplot.append(list(a.trapz[a.rock_type == t].values))
        toplot.append(markers[i])
    plt.plot(*toplot, markersize=8)
    plt.legend(tr)
    plt.gca().set_xlabel('pH (measurement in field %s)' % ph_df.date.max())
    plt.gca().set_ylabel('$\int_{t_0}^{t_1} \mathrm{flux\  dt}$')
    plt.grid(True)


if experiment.name in ['migmin', 'buckets']:
    plt.cla()
    plot_ph_vs_flux(trapz_df(df), ph_df)
    plt.show()
    
# %% plot_date
    
def plot_date(df,date,sort_by=['mixture', 'fertilizer', 'rock_type']):
    justdate = df.date.map(lambda x:x[:8])
    d = df[justdate==date]
    d = d.sort_values(by=sort_by)
    y = d.N2O_slope
    plt.bar(range(len(y)), y)
    plt.gca().set_xticks(range(len(y)))
    xlabs = []
    for i in d.index:
        di = d.loc[i]
        xlabs.append(repr(di.plot_nr) + ' ' + ' '.join([di[s] for s in sort_by]))
    plt.gca().set_xticklabels(xlabs, rotation=60)
    return d

if experiment.name in ['migmin', 'buckets']:
    treatments = ['rock_type']
else:
    treatments = ['mixture', 'fertilizer', 'rock_type']
    
#plot_date(df,'20160923')
print('For the last day:')
#d = plot_date(df, df.date.max(), treatments)
#d = plot_date(df,'20160923', treatments)
#print(d[['N2O_N_mug_m2h', 'plot_nr'] + treatments])
# %% subplots integration gothrough

# %% ginput-examples
# 
# This makes the plot come up in a separate window, where they can
# be clicked on. (The plot window sometimes shows up behind the spyder window.)

# We want to
# 1: have a map with dots where we have sampled, and click on the dots to
#    get the info about the sample and see the regression
# 
# ginput_show.ginput_check_points(df, rectangles, regr)
#
# 2: have a plot of results (slopes) and click on the dots to see the same
#
# ginput_show.ginput_check_regres(df, regr)
#

# %% Making your own regression function:

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
        regressions = dict()
        regressions['left'] = dict()
        regressions['right'] = dict()
        regressions['left']['CO2'] = reg(co2['left'][0], co2['left'][1])
        regressions['left']['N2O'] = reg(n2o['left'][0], n2o['left'][1])
        regressions['right']['CO2'] = reg(co2['right'][0], co2['right'][1])
        regressions['right']['N2O'] = reg(n2o['right'][0], n2o['right'][1])
        return regressions

# Then you can try this:
# regr2 = MyRegressor('another_slopes_filename', {'p1': 100, 'p2': 3.14})
# data = get_data.get_file_data(os.path.join(resdir.raw_data_path, example_file))
# reg = regr2.find_all_slopes(data, plotfun=plt.plot)
# print(reg)

print("""
      
Commands you may want to try:
    
df2 = df[(df.date>'20171030')&(df.date<'20181010')]
plot_all(df2)
plt.show()
a, b = barplot_trapz(df2, True)
plt.show()
plt.clf();plt.subplot(1,1,1)
plot_ph_vs_flux(trapz_df(df), ph_df)
plt.show()
model = 'np.log(trapz + 0.005) ~ C(rock_type)'
ols_trapz_res = ols(model, data=trapz_df(df2)).fit()
print(ols_trapz_res.summary())
ginput_show.ginput_check_points(df2, rectangles, regr)
ginput_show.ginput_check_regres(df, regr)
ginput_check_regres(df[df.plot_nr==1])


""")

