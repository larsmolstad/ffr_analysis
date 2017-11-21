"""

Showing some examples of how to deal with the ffr data

"""


#%% Imports:
import os
import glob
from plotting_compat import plt
import numpy as np
import pandas as pd


# for spyder: os.chdir('c:/zip/sort_results/sort_ffr_results/')
import resdir
import get_data
import utils
import find_regressions as fr
import plot_rectangles as pr
import sort_results as sr
import find_plot
import weather_data
import flux_calculations


#%% Override the default result directories:
resdir.raw_data_path = 'c:\\zip\\sort_results\\results'
resdir.slopes_path = 'c:\\zip\\sort_results'
example_file = '2016-06-16-10-19-50-x599234_725955-y6615158_31496-z0_0-h0_743558650162_both_Plot_9_'


#%% Never mind this:

do_show = False
do_wait = False
def show_and_wait():
    if do_show:
        plt.show()
    if do_wait:
        raw_input('Press Enter ')

        
#%% Get a list of all result files
filenames = glob.glob(os.path.join(resdir.raw_data_path, '*'))
print("number of files: %d"%len(filenames))


# Get the data from file number 1000
a = get_data.get_file_data(filenames[1000])
plt.cla()
plt.plot(a['N2O'][0], a['N2O'][1], '.')

# example_file has some fluxes:
filename = os.path.join(resdir.raw_data_path, example_file)
a = get_data.get_file_data(filename)
plt.plot(a['N2O'][0], a['N2O'][1], '.')
show_and_wait()


#%% simplify working on the repl (command line)
b = utils.dict2inst(a)
print(dir(b))
# (now you can do b.N2O etc with tab completion)


#%% Do a regression:
plt.cla()
a = get_data.get_file_data(filename)
reg = fr.find_all_slopes(a, interval=100, co2_guides=True, plotfun=plt.plot)
fr.print_reg(reg)
show_and_wait()
# Here, a can be the filename or the data dict. Interval is the length
# of time of the regression line. If co2_guides==True, the interval in
# time where the co2 curve is the steepest is used for the time of
# regression for the N2O. Otherwise, the steepest part of the N2O is used.
# (todo mse etc)


# (the division of the data from the two chambers is done in divide_left_and_right.py like this)
import divide_left_and_right
ad = divide_left_and_right.group_all(a)


#%% Plot the rectangles of migmin
plt.cla()
plt.hold(True)
rectangles = pr.migmin_field_rectangles()
pr.plot_rectangles(rectangles.values(), rectangles.keys())
show_and_wait()


#%% Sort results according to the rectangles, put them in a Pandas dataframe
pd.set_option('display.width', 250)
# The slopes have been stored in slopes3.txt with the command
# python find_regressions.py --out slopes3.txt ../results
# slopes3.txt has the slopes from all the files in the results folder.
# make_df_from_slope_file picks the one that are inside rectangles
name = os.path.join(resdir.slopes_path, 'slopes3.txt')
rectangles = pr.migmin_field_rectangles()

df, df0 = sr.make_df_from_slope_file(name, rectangles,
                                     find_plot.treatments,
                                     remove_redoings_time=3600)

print(df)

#%% Pandas... Pandas is a python library that gives python R-like
# dataframes. It takes some time to learn Pandas, although there is an
# introduction called "10 minutes to Pandas"
print(df.head())
print(df.tail())
print(df.columns)
d = df[df.plot_nr==1]
plt.cla()
plt.axis('auto')
plt.plot(d['t'], d['N2O'])
print d['N2O']


#%% add in some weather data, calculate fluxes, wrap it up in a function:
# (if you have internet, you can do weather_data.data.update() first)

def update(precip_dt=2, rectangles=rectangles):
    df, df0 = sr.make_df_from_slope_file(name, rectangles,
                                         find_plot.treatments,
                                         remove_redoings_time=3600)
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    df['precip2'] = weather_data.data.get_precip2(df.t, [0, precip_dt])
    df['N2O_N_mmol_m2day'] = 2 * 1000 * 86400 * \
        flux_calculations.calc_flux(df.N2O, df.Tc)  # 2 because 2 N in N2O
    df['CO2_C_mmol_m2day'] = 1000 * 86400 * \
        flux_calculations.calc_flux(df.CO2, df.Tc)
    return df, df0

df, df0 = update()

print df.head()


#%% A little check that the sorting is ok:

def test_nr(nr):
    pr.plot_rectangles(rectangles.values(), rectangles.keys())
    d = df0[df0.plot_nr==nr]
    plt.plot(d.x, d.y, '.')

nrs = np.unique(df[df.treatment=='Norite'].plot_nr)

print nrs

plt.cla()

for nr in nrs:
    test_nr(nr)

#%% Just the days with high fluxes:

df2 = sr.filter_for_average_slope_days(df, 0.0005)

# useful:
a = df.groupby('daynr').N2O
a = a.mean().values[a.count().values>10]
plt.hist(a*1000, bins='auto')

#%%barmaps

_ = sr.barmap_splitted(df, df0, theta=0)

#%%
plt.show()
raw_input('')
