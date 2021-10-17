"""
 Sigrids capture experiments
"""
 
# First, select this file's directory in the white bar up to the
# right. You can do this by right-clicking on ".py" above and choosing
# "Set console working directory"
 
# The button that runs current "cell" executes the code highlighted
# (Cells are code between lines starting with # %%)
  
# %% Imports:
import sys
import os
import glob
from collections import namedtuple
import numpy as np
import pylab as plt
import pandas as pd
pd.options.mode.chained_assignment = None
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), '../../prog')))
import resdir
import get_data
import utils
import find_regressions
import sort_results as sr
import weather_data
import flux_calculations
import polygon_utils
# import ginput_show
# import textwrap
# import regression
# import divide_left_and_right
# from polygon_utils import plot_rectangles
# import scipy.stats
# from statsmodels.formula.api import ols#, rlm
# from statsmodels.stats.anova import anova_lm
# import statsmodels.api as sm
# from scipy.stats import norm
# import xlwt
#import shutil
#import errno
 
fixpath = utils.ensure_absolute_path
 
start_date = '2021-08-19'
stop_date =  '2099-01-01'  #YYYYMMDD  stop_date has to be one day after the last date you want
redo_regressions =  False
 
options = {'interval': 100,
           'start':0,
           'stop':180,
           'crit': 'steepest',
           'co2_guides': True,
           'correct_negatives':False
           }
 
save_options= {'show_images':True,
               'save_images':True,
               'save_detailed_excel':True,
               'sort_detailed_by_experiment':True
               }
 
remove_redoings_time = 10 #seconds
 
# flux_units = {'N2O': {'name': 'N2O_N_mmol_m2day', 'factor': 2 * 1000 * 86400},
#              'CO2': {'name': 'CO2_C_mmol_m2day', 'factor': 1000 * 86400}}
flux_units = {'N2O': {'name': 'N2O_N_mug_m2h', 'factor': 2 * 14 * 1e6 * 3600},
              'CO2': {'name': 'CO2_C_mug_m2h', 'factor': 12 * 1e6 * 3600}}
 
specific_options_filename = fixpath('specific_options.xls')
 
resdir.raw_data_path = fixpath('raw_data')
detailed_output_path = fixpath('output/detailed_regression_output_unsorted')
find_regressions.make_detailed_output_folders(detailed_output_path)
 
excel_filename_start = "output/capture"
slopes_filename = fixpath("output/capture_slopes.txt")
 
# Finding the raw data files
all_filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
print("number of measurement files from robot: %d" % len(all_filenames))
# %%


def position(filename):
    a = get_data.parse_filename(filename)['vehicle_pos']
    return np.array([a['x'], a['y']])

positions = [position(name) for name in all_filenames]
x = np.array([x[0] for x in positions])
y = np.array([x[1] for x in positions])
offset = namedtuple('Point', ('x', 'y'))(x=5.99201e5, y=6.615259e6)
plt.scatter(x-offset.x, y-offset.y, marker='.')

#--
def file_belongs(filename):
    name = os.path.split(filename)[1]
    date_ok = start_date <= name.replace('-','') <= stop_date
    x, y = position(filename)
    pos_ok = 0 < x - offset.x < 45 and 0 < y - offset.y < 55 
    #text_ok = name.find('Measure') > -1
    return date_ok and pos_ok

filenames = [x for x in all_filenames if file_belongs(x)]
print('number of measurement files included in this run:', len(filenames))
 
filenames.sort() # alphabetically (i.e., by date)
 
# Make the "regressor object" regr which will be used further below.
# It contains the functions and parameters (options) for doing the regressions.
 
regr = find_regressions.Regressor(slopes_filename, options, save_options,
                                  specific_options_filename, detailed_output_path)
 
if not os.path.isfile(slopes_filename):
    open(slopes_filename, 'a').close() #creates the file

if redo_regressions:
    regr.find_regressions(filenames)
else:
    regr.update_regressions_file(filenames) #updates resfile

 
# plot_error_number(n, key='N2O'):
 
#%%
"""
Preparing the data for "RegressionOutput" Excel export
"""
# %%
#Sort results according to the rectangles, put them in a Pandas dataframe
# Read "10 minutes to pandas" to understand pandas (or spend a few hours)
 
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
 
df = sr.make_simple_df_from_slope_file(slopes_filename)
 
df.sort_values('date', inplace=True)
#--
# plt.ion(); plt.cla()
# plt.scatter(df.x-offset.x, df.y-offset.y, s=1)
# plt.scatter(x-offset.x, y-offset.y, color="red", s=1)


rect1 = polygon_utils.Polygon(0, 0, W=37.5, L=48)
rect1.rotate(.4152).move(15.2,-2.55)
#rect1.rotate(.4152).move(599216.2,6615256.5)

rectangles = rect1.grid_rectangle(6,6)

# polygon_utils.plot_rectangles(rectangles, textkwargs={'fontsize': 5}, linewidth=.1)

df['nr'] = [polygon_utils.find_polygon(p[0]-offset.x, p[1]-offset.y, rectangles) + 1
            for p in  zip(df.x, df.y)]

    
treatmentlist = [( 1,  6), ( 2,  2), ( 3, 12), ( 4,  7), ( 5,  4), ( 6,  9),
                 ( 7, 11), ( 8, 10), ( 9,  1), (10,  5), (11,  8), (12,  3),
                 (13, 12), (14, 11), (15,  4), (16, 10), (17,  5), (18,  2),
                 (19,  1), (20,  7), (21,  8), (22,  6), (23,  3), (24,  9),
                 (25,  3), (26, 10), (27,  2), (28,  4), (29,  5), (30,  1),
                 (31, 11), (32,  6), (33, 12), (34,  9), (35,  8), (36,  7)]

treatments = {x[0]:x[1] for x in treatmentlist}

df['treatment'] = [treatments[i] for i in df.nr]

plt.cla()

polygon_utils.plot_rectangles(rectangles, textkwargs={'fontsize': 5}, linewidth=.1)

colors = 'bgrcmyk'*10
markers = '.......xxxxxxx'*5

for t in sorted(set(df.treatment)):
    d = df[df.treatment==t]
    plt.scatter(d.x-offset.x, d.y-offset.y, s=10, color=colors[t-1], marker=markers[t-1])


 
def finalize_df(df, precip_dt=2):
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    df['N2O_mol_m2s'] = flux_calculations.calc_flux(df.N2O_slope, df.Tc)
    df['CO2_mol_m2s'] = flux_calculations.calc_flux(df.CO2_slope, df.Tc)
    Nunits = flux_units['N2O']
    Cunits = flux_units['CO2']
    df[Nunits['name']] = Nunits['factor'] *  df.N2O_mol_m2s
    df[Cunits['name']] = Cunits['factor'] *  df.CO2_mol_m2s
    df = sr.rearrange_df(df)
    return df
 
df = finalize_df(df)
df['days'] = (df.t - min(df.t))/86400

print('from ', df.date.min())
print('to   ', df.date.max())

openthefineapp = False
excel_filenames = [fixpath(excel_filename_start + '_' + s + '.xls')
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
 
 

# _slopes and _all_columns are additional output files with regression results sorted by date
print(flux_units['N2O']['name'])
tokeep = ['t', 'date', 'days', 'nr', 'side', 'treatment',
          flux_units['N2O']['name'], flux_units['CO2']['name'],
          'N2O_slope', 'CO2_slope', 'filename']
df2 = df[tokeep]
try:
    df2.to_excel(excel_filenames[1])
except:
    print("didn't work todo")
 

def plot_something(df, key, value, what="N2O", **kwargs):
    d = df[df[key]==value]
    kwargs = {**{'linewidth': .5, 'marker':'.', 'markersize':2}, **kwargs}
    plt.plot(d.days, d.N2O_N_mug_m2h if what=="N2O" else d.CO2_C_mug_m2h,
             label=value, **kwargs)
    
def plot_treatment(df, treatment, what="N2O", **kwargs):
    plot_something(df, 'treatment', treatment, what, **kwargs)

def plot_nr(df, nr, what="N2O", **kwargs):
    plot_something(df, 'nr', nr, what, **kwargs)

    
plt.cla()

for x in sorted(set(treatments.values())):
    plot_treatment(df, x )

plt.legend()

plt.show()

plt.cla();



# todo ta med trapz og plott fra cookbook

# print(df[['t', 'x', 'y', 'nr']].head(10))
# print(df[df.treatment == 1][['t', 'x', 'y', 'nr', 'N2O_slope']].head(10))
# pnr = df.nr.values[0]
# print('plotting N2O slopes for plot_nr', pnr)
# d = df[df.nr == pnr]
# plt.cla()
# plt.axis('auto')
# plt.plot(d['t'], d['N2O_slope'], '.-')
# plt.show()
# print(d['N2O_slope'].tail())
# print(df.head())
 
