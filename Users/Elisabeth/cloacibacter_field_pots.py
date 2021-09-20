elisabeth.hiis@nmbu.no
cloacibacter_field_pots
Elisabeth Gautefall Hiis
Wed 9/8/2021 9:56 AM
"""
Elisabeth's bucket experiment
 
"""
 
# First, select this file's directory in the white bar up to the
# right. You can do this by right-clicking on ".py" above and choosing
# "Set console working directory"
 
# The button that runs current "cell" executes the code highlighted
# (Cells are code between lines starting with # %%)
 
 
# %% Imports:
import sys
import os
import time
import glob
import numpy as np
import pylab as plt
import pandas as pd
import textwrap
pd.options.mode.chained_assignment = None
sys.path.append(os.path.realpath(
    os.path.join(os.getcwd(), '../../prog')))
import resdir
import get_data
import utils
import regression
import find_regressions
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
import xlwt
import shutil
import errno
import re
import datetime
current_path = os.getcwd()
 
fixpath = utils.ensure_absolute_path
# %% #################  EDIT THESE PARAMETERS: ################################
#
 
"""Explanation"""
# start_date and stop_date
#    The first and last date of the raw data that will be considered
#    Example '20171224' Note that since stopdate is compared
#    to a string which includes hours, minutes and seconds, the stop_date '20180305'
#    will not be included unless written as e.g., '2018030524' (or just '20180306')
 
# redo_regressions:
#    If True, discard all the regressions in the slopes_file and do
#    them over. If False, redo only if options have changed (given
#    either in the options variable or the specific_options spreadsheet)
#    Note:  detailed raw data and image file are only created if that regression is run.
#
# options:
#    Set the parameters applied to all measurements unless specified in specific_options_filename.
#    Here, the options have: 'interval': the length of the
#    regression interval (seconds) 'crit': how to choose the best
#    interval within the run (for example the best 100 seconds within 180
#    seconds). 'crit' can be 'steepest' or 'mse' (mean squared error)
#    'co2_guides': wether or not the N2O regression will be done on the
#    same interval as the CO2 regression. Se the excel specific_options file
#    for more options
#    'correct_negatives':
#
# remove_redoings_time:
#    Time in seconds, which if more than one measurement is found for the same plot #, it keeps only the FIRST one.
#    Beware especially if there are no plot numbers defined - may want to keep remove_redoings_time low.
#
# flux_units:
#    The units to calculate CO2 and N2O flux in the RegressionOutput file.
#    factor is the conversion factor from mol/s/m2 to the given unit.
#
# FILE PATHS
# *NOTE:  Directories must exist before the program is run.
# specific_options_filename:
#    Full file path of a spreadsheet containing exceptions to override the default regression options
#    Or, if only a filename is used, it looks in the program's working directory
#
# resdir.raw_data path:
#    Directory where raw measurements imported from the robot are stored
#    Examples:  '../../_RAWDATA' or Y:/Shared/N-group/FFR/_RAWDATA
#
# reg_output_path:
#    Directory to save the RegressionOutput Excel file
#
# detailed_output_path:
#    Directory to save raw data details and .png images of each measurement's regression points and line
#
#######################################################################
 
start_date = '20210713'
stop_date = '20210901'  #YYYYMMDD  stop_date has to be one day after the last date you want
redo_regressions =  False
 
options = {'interval': 100,
           'start':0,
           'stop':180,
           'crit': 'steepest',
           'co2_guides': True,
           'correct_negatives':False
           }
 
save_options= {'show_images':False,
               'save_images':False,
               'save_detailed_excel':True,
               'sort_detailed_by_experiment':True
               }
 
remove_redoings_time = 10 #seconds
 
# remove_data_outside_rectangles = False
 
# flux_units = {'N2O': {'name': 'N2O_N_mmol_m2day', 'factor': 2 * 1000 * 86400},
#              'CO2': {'name': 'CO2_C_mmol_m2day', 'factor': 1000 * 86400}}
flux_units = {'N2O': {'name': 'N2O_N_mug_m2h', 'factor': 2 * 14 * 1e6 * 3600},
              'CO2': {'name': 'CO2_C_mug_m2h', 'factor': 12 * 1e6 * 3600}}
 
 
specific_options_filename = fixpath('specific_options.xls')
 
resdir.raw_data_path = fixpath('raw_data')
                                     
detailed_output_path = fixpath('detailed_regression_output_unsorted')
find_regressions.make_detailed_output_folders(detailed_output_path)
 
excel_filename_start = "superbug_pots"
slopes_filename = fixpath("superbug_pots_slopes.txt")
 
# Finding the raw data files
all_filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
print("number of measurement files from robot: %d" % len(all_filenames))
 
def file_belongs(filename):
    name = os.path.split(filename)[1]
    date_ok = start_date <= name.replace('-','') <= stop_date
    text_ok = name.find('Measure') > -1
    return date_ok and text_ok
 
filenames = [x for x in all_filenames if file_belongs(x)]
print('number of measurement files included in this run:', len(filenames))
 
filenames.sort() # alphabetically (i.e., by date)
 
# Make the "regressor object" regr which will be used further below.
# It contains the functions and parameters (options) for doing the regressions.
 
regr = find_regressions.Regressor(slopes_filename, options, save_options,
                                  specific_options_filename, detailed_output_path)
 
if redo_regressions:
    regr.find_regressions(filenames)
else:
    regr.update_regressions_file(filenames) #updates resfile
 
# plot_error_number(n, key='N2O'):
 
#%%
"""
Preparing the data for "RegressionOutput" Excel export
"""
# %% Sort results according to the rectangles, put them in a Pandas dataframe
# Read "10 minutes to pandas" to understand pandas (or spend a few hours)
 
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
 
df = sr.make_simple_df_from_slope_file(slopes_filename)
 
df.sort_values('date', inplace=True)
 
def station_nr(filename):
    """return the number at the end of the raw data filename"""
    I = filename.find('Measure')
    return int(re.findall('\d+', filename[I:])[0])
 
df['nr'] = [station_nr(x) for x in df.filename]
 
treatments = {
    (1, 'left') : 'cloaci live',
    (1, 'right') : 'cloaci live',
    (2, 'left') : 'cloaci dead',
    (2, 'right') : 'cloaci dead',
    (3, 'left') : 'live dig',
    (3, 'right') : 'live dig',
    (4, 'left') : 'water',
    (4, 'right') : 'water',
    (5, 'left')  : 'cloaci live',
    (5, 'right') : 'cloaci live',
    (6, 'left') : 'cloaci dead',
    (6, 'right') : 'cloaci dead',
    (7, 'left') : 'live dig',
    (7, 'right') : 'live dig',
    (8, 'left') : 'water',
    (8, 'right') : 'water',
    (9, 'left') : 'cloaci live',
    (9, 'right') : 'cloaci live',
    (10, 'left') : 'cloaci dead',
    (10, 'right') : 'cloaci dead',
    (11, 'left') : 'live dig',
    (11, 'right') : 'live dig',
    (12, 'left') : 'water',
   (12, 'right') : 'water',
    (13, 'left') : 'cloaci live',
    (13, 'right') : 'cloaci live',
    (14, 'left') : 'cloaci dead',
    (14, 'right') : 'cloaci dead',
    (15, 'left') : 'live dig',
    (15, 'right') : 'live dig',
    (16, 'left') : 'water',
    (16, 'right') : 'water'
    }
 
df['treatment'] = [treatments[(x[1].nr, x[1].side)] for x in df.iterrows()]
# x[0] is index, x[1] is the rest
 
# not really neccessary
df = df[(df.date >= start_date) & (df.date <= stop_date)]
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
 
weather_data.data.update()
 
def finalize_df(df, precip_dt=2):
   bucket_factor = (50/23.5)**2 * 0.94
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    df['N2O_mol_m2s'] = flux_calculations.calc_flux(df.N2O_slope, df.Tc)
    df['CO2_mol_m2s'] = flux_calculations.calc_flux(df.CO2_slope, df.Tc)
    df.N2O_mol_m2s *= bucket_factor
    df.CO2_mol_m2s *= bucket_factor
    Nunits = flux_units['N2O']
    Cunits = flux_units['CO2']
    df[Nunits['name']] = Nunits['factor'] *  df.N2O_mol_m2s
    df[Cunits['name']] = Cunits['factor'] *  df.CO2_mol_m2s
    df = sr.rearrange_df(df)
    return df
 
df = finalize_df(df)
 
# print(df.head())
 
openthefineapp = False
excel_filenames = [fixpath('../../'+excel_filename_start + '_' + s + '.xls')
                   for s in 'RegressionOutput slopes all_columns'.split()]
 
# First, the main RegressionOutput file
try:
    df.to_excel(excel_filenames[0])
    print('Regression Output file(s) written to parent directory')
    if openthefineapp:
        os.system(excel_filenames[0])
except:
    print('Regression Output file(s) NOT written -- was it open?')
    pass
 
 
 
# _slopes and _all_columns are dditional output files with regression results sorted by date
print(flux_units['N2O']['name'])
tokeep = ['t', 'date', 'nr', 'side', 'treatment',
          flux_units['N2O']['name'], flux_units['CO2']['name'],
          'N2O_slope', 'CO2_slope', 'filename']
df2 = df[tokeep]
df2['days'] = (df.t - min(df.t))/86400
try:
    df2.to_excel(excel_filenames[1])
except:
    print("didn't work todo")
 
 
def plot_treatment(treatment, what="N2O"):
    df = df2[df2.treatment==treatment]
    plt.plot(df.days, df.N2O_N_mug_m2h if what=="N2O" else df.CO2_C_mug_m2h,
             label=treatment)
 
plt.cla()
plot_treatment('water')
plot_treatment('live dig')
plot_treatment('cloaci dead')
plot_treatment('cloaci live')
plt.legend()
plt.show()
 
#plt.cla()
#plot_treatment('water',1)
#plot_treatment('p.stutzeri',1)
#plot_treatment('cloacibacter',1)
#plt.show()
 
water = df2[df2.treatment=='water']
livedig = df2[df2.treatment=='live dig']
dead = df2[df2.treatment=='cloaci dead']
live = df2[df2.treatment=='cloaci live']
 
print("live mean/dead mean:")
print(live.mean()/dead.mean())
#print(p.mean()/cl.mean())
 
# todo ta med trapz og plott fra cookbook
# og detailed regression output files
 
 
# plot_raw(filename, key=["N2O"])
 
# examplefilename = with_raw_dir(example_file)
 
# Do a single regression:
# plt.cla()
###reg = regr.find_all_slopes(examplefilename, do_plot=True)
###plt.show()
# Print the slopes, intercepts, mse etc:
###find_regressions.print_reg(reg)
# another way:
# data = get_data.get_file_data(filename)
# reg = regr.find_all_slope(data)
 
#%%
def plot_everything():
    for i, treatment in enumerate(sorted(set(treatments.values()))):
        plt.subplot(2,2,i+1, title=treatment)
        df3 = df2[df2.treatment==treatment]
        for nr in sorted(set(df3.nr)):
            df4 = df3[df3.nr==nr]
            for side in sorted(set(df4.side)):
                df = df4[df4.side==side]
                print((i, treatment, nr, side))
                plt.plot(df.days, df.N2O_N_mug_m2h, '.-', label=side+repr(nr))
            plt.legend()
            plt.ylim([0, 120000])
 
 
plot_everything()
#%%
 
name='C:/Users/elhi/OneDrive - Norwegian University of Life Sciences/superbug/moisture_temp.xlsx'
a = pd.read_excel(open(name, 'rb'),  sheet_name='moist_temp') 
a=a[2:]
 
def datetime_toepoch(d):
    return (d - datetime.datetime(1970,1,1)).total_seconds()
import bisect
 
def find_closest_sorted(df, name, x): #todo interpolate and
    i = bisect.bisect_left(df[name].values, x)
    if i==len(df[name]) or \
        np.abs(df[name].iloc[i] - x) > np.abs(df[name].iloc[i-1] - x):
        i -= 1
    return i
 
 
a['time'] = a['Bucket'].apply(datetime_toepoch)
 
translations = (('Port 1',  'moist_1'),
                ('Port 1.1', 'temp_1'),
                ('Port 2',  'moist_2'),
                ('Port 2.1', 'temp_2'),
                ('Port 3',  'moist_3'),
                ('Port 3.1', 'temp_3'),
                ('Port 4',  'moist_4'),
                ('Port 4.1', 'temp_4'))
 
a.rename(columns=dict(translations), inplace=True)
 
a.drop(['Port 1.2', 'Port 2.2', 'Port 3.2', 'Port 4.2',
        'Port 5', 'Port 5.1', 'Bucket'],
       1, inplace=True)
 
names = [x[1] for x in translations]
#%%
for s in names:
    df2[s] = np.nan
 
df2['logger_time'] = np.nan
 
for i in range(len(df2)): #todo finne ordentlig maate
    j = find_closest_sorted(a, 'time', df2.iloc[i].t)
    for s in names:
        df2[s].iloc[i] = a[s].iloc[j]
    df2.logger_time.iloc[i] = a.time.iloc[j]
 
#%%
def plot_everything2():
    for i, treatment in enumerate(sorted(set(treatments.values()))):
        plt.subplot(2,2,i+1)
        plt.cla()
        plt.title(treatment)
        df3 = df2[df2.treatment==treatment]
        for nr in sorted(set(df3.nr)):
            df4 = df3[df3.nr==nr]
            for side in sorted(set(df4.side)):
                df = df4[df4.side==side]
                print((i, treatment, nr, side))
                plt.plot(df.days, df.N2O_N_mug_m2h, '.-', label=side+repr(nr))
            plt.legend()
            plt.ylim([0, 120000])
            for x in names[::2]:
                plt.plot(df2.days, (df2[x]-0.298)*10000, '.',color="blue", markersize=1)
            for x in names[1::2]:
                plt.plot(df2.days, df2[x]*100, '.--', color="red", markersize=1)
            plt.grid(True)
 
plot_everything2()
plt.tight_layout()
 
#%%
 
df3 = df2.sort_values(['nr', 'side', 't'])
df3.to_excel('fluxes_and_logdata.xls')
