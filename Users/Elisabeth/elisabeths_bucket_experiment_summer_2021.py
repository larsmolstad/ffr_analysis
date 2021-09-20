"""
Elisabeths bucket experiment

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

#import buckets as experiment
start_date = '20210604'
stop_date = '20210901'  #YYYYMMDD  stop_date has to be one day after the last date you want
redo_regressions =  True

options = {'interval':100,
           'start':0,
           'stop':180,
           'crit': 'steepest',
           'co2_guides': True,
           'correct_negatives':False
           }

save_options= {'show_images':False,
               'save_images':False,#True,
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
                                     
detailed_output_path = fixpath('Detailed_regression_output_Unsorted')
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

df = sr.make_simple_df_from_slope_file(slopes_filename)

df.sort_values('date', inplace=True)

def station_nr(filename):
    """return the number at the end of the raw data filename"""
    I = filename.find('Measure')
    return int(re.findall('\d+', filename[I:])[0])

df['nr'] = [station_nr(x) for x in df.filename]

treatments = {
    (1, 'left') : 'control',
    (1, 'right') : 'control',
    (2, 'left') : 'cloacibacter',
    (2, 'right') : 'cloacibacter',
     (3, 'left')  : 'p.stutzeri',
    (3, 'right') : 'cloacibacter',
    (4, 'left') : 'p.stutzeri',
    (4, 'right') : 'p.stutzeri'
    }

df['bct'] = [(x[1].nr, x[1].side) for x in df.iterrows()]
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
tokeep = ['date', 'nr', 'side', 'treatment',
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
    plt.plot(df.days, df.N2O_N_mug_m2h if what=="N2O" else df.CO2_C_mug_m2h)

plt.cla()
plot_treatment('control')
plot_treatment('p.stutzeri')
plot_treatment('cloacibacter')
plt.show()

plt.cla()
plot_treatment('control',1)
plot_treatment('p.stutzeri',1)
plot_treatment('cloacibacter',1)
plt.show()

c = df2[df2.treatment=='control']
cl = df2[df2.treatment=='cloacibacter']
p = df2[df2.treatment=='p.stutzeri']

print(p.mean()/c.mean())
print(p.mean()/cl.mean())

# todo ta med trapz og plott fra cookbook
# og detailed regression output files

def with_raw_dir(filename):
    return os.path.join(resdir.raw_data_path, filename)

def plot_raw(examplefilename, key='N2O'):
    "key may be 'N2O', 'CO2', 'Wind'"
    if not os.path.isfile(examplefilename):
        examplefilename = with_raw_dir(examplefilename)
    a = get_data.get_file_data(examplefilename)
    plt.plot(a[key][0], a[key][1], '.')
    plt.gca().set_xlabel('seconds')
    if key in ['N2O', 'CO2', 'CO']:
        plt.gca().set_ylabel('ppm')
    return a

def plot_iloc(df, i):
    print(df.iloc[i].filename)
    plot_raw(df.iloc[i].filename)
    plt.show()
    

def regnr(df, i):
    filename = df.iloc[i].filename
    print(filename)
    plt.cla()
    r = regr.find_all_slopes(with_raw_dir(filename), do_plot=True)
    plt.show()
    return r
    
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

