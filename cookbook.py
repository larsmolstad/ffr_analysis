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
from polygon_utils import plot_rectangles
import sort_results as sr
import find_plot
import weather_data
import flux_calculations


#%%

################### EDIT THESE:

# Override the default result directories:
resdir.raw_data_path = 'c:\\zip\\sort_results\\results'
resdir.slopes_path = 'c:\\zip\\sort_results'
#resdir.slopes_path = 'c:/users/larsmo/downloads'

# Select which rectangles and treatments you want:
import migmin
rectangles = migmin.migmin_rectangles()
treatments = migmin.treatments
# or
#rectangles = something.agropro_rectangles()

#################### END EDIT THESE


example_file = '2016-06-16-10-19-50-x599234_725955-y6615158_31496-z0_0-h0_743558650162_both_Plot_9_'


# Plotting the rectangles 
plt.cla()
plt.hold(True)
plot_rectangles(rectangles)
# with treatments:
plt.cla()
keys = list(rectangles)
r = [rectangles[k] for k in keys]
tr = [treatments[k] for k in keys]
plot_rectangles(r, tr)


#%% plotting examples
x = [1,2,3,4]
y = [1,2,2,1]
plt.plot(x,y)
plt.show()
plt.axis('equal')
plt.axis('auto')
s1 = plt.subplot(2,3,1)
x = np.linspace(0, 3*np.pi)
plt.plot(x, np.sin(x))

#%% 
plt.clf()
plt.subplot(1,1,1)

        
#%% Get a list of all result files
filenames = glob.glob(os.path.join(resdir.raw_data_path, '*'))
print(("number of files: %d"%len(filenames)))


# Get the data from file number 1000
a = get_data.get_file_data(filenames[1000])
plt.cla()
plt.plot(a['N2O'][0], a['N2O'][1], '.')

# example_file has some fluxes:
filename = os.path.join(resdir.raw_data_path, example_file)
a = get_data.get_file_data(filename)
plt.cla()
plt.plot(a['N2O'][0], a['N2O'][1], '.')
plt.show()

#%% simplify working on the repl (command line)
b = utils.dict2inst(a)
print((dir(b)))
# (now you can do b.N2O etc with tab completion)


#%% Do a regression:
plt.cla()
a = get_data.get_file_data(filename)
reg = fr.find_all_slopes(a, interval=100, co2_guides=True, plotfun=plt.plot)
fr.print_reg(reg)
plt.show()
# Here, a can be the filename or the data dict. Interval is the length
# of time of the regression line. If co2_guides==True, the interval in
# time where the co2 curve is the steepest is used for the time of
# regression for the N2O. Otherwise, the steepest part of the N2O is used.
# (todo mse etc)


# (the division of the data from the two chambers is done in divide_left_and_right.py like this)
import divide_left_and_right
ad = divide_left_and_right.group_all(a)

#%% Do many regressions (more instructions will follow, but see README.txt) 

resfile = os.path.join(resdir.slopes_path, 'slopes3.txt')

# this takes a long time, so I commented it out:

# write regressions to resfile:
#fr.find_regressions(resdir.raw_data_path, resfile, 100, True) 

# update resfile without redoing regressions:
#fr.update_regressions_file(resdir.raw_data_path, resfile, 100, True)

#%% Sort results according to the rectangles, put them in a Pandas dataframe
pd.set_option('display.width', 250)
# The slopes have been stored in slopes3.txt with the command
# python find_regressions.py --out slopes3.txt ../results
# slopes3.txt has the slopes from all the files in the results folder.
# make_df_from_slope_file picks the one that are inside rectangles
name = os.path.join(resdir.slopes_path, 'slopes3.txt')

df, df0 = sr.make_df_from_slope_file(name, rectangles,
                                     treatments,
                                     remove_redoings_time=3600)

print(df)

#%% Pandas... Pandas is a python library that gives python R-like
# dataframes. It takes some time to learn Pandas, although there is an
# introduction called "10 minutes to Pandas"
print((df.head()))
print((df.tail()))
print((df.columns))
pnr = df.plot_nr.values[0]
print('plotting N2O slopes for plot_nr', pnr)
d = df[df.plot_nr==pnr]
plt.cla()
plt.axis('auto')
plt.plot(d['t'], d['N2O'])
print(d['N2O'])
plt.show()

#%% add in some weather data, calculate fluxes, wrap it up in a
# function. (if you have internet, you can do
# weather_data.data.update() first. This will download weather data
# from yr and save them)

def update(precip_dt=2, rectangles=rectangles, treatments=treatments):
    df, df0 = sr.make_df_from_slope_file(name,
                                         rectangles,
                                         treatments,
                                         remove_redoings_time=3600)
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    df['precip2'] = weather_data.data.get_precip2(df.t, [0, precip_dt])
    df['N2O_N_mmol_m2day'] = 2 * 1000 * 86400 * \
        flux_calculations.calc_flux(df.N2O, df.Tc)  # 2 because 2 N in N2O
    df['CO2_C_mmol_m2day'] = 1000 * 86400 * \
        flux_calculations.calc_flux(df.CO2, df.Tc)
    df = sr.rearrange_df(df)
    return df, df0

df, df0 = update()

print(df.head())


#%% A little check that the sorting is ok:

def test_nr(nr):
    plot_rectangles(list(rectangles.values()), list(rectangles.keys()))
    d = df[df.plot_nr==nr]
    plt.plot(d.x, d.y, '.')
    plt.show()
    
nrs = np.unique(df[df.treatment=='Norite'].plot_nr)

print(nrs)

plt.cla()

for nr in nrs:
    test_nr(nr)

#%% Just the days with high fluxes:

df2 = sr.filter_for_average_slope_days(df, 0.0005)

# useful:

a = df.groupby('daynr').N2O
a = a.mean().values[a.count().values>10]
plt.cla()
plt.hist(a*1000, bins='auto')

#%%barmaps

#_ = sr.barmap_splitted(df, theta=0)

#%%Excel

df.to_excel('..\excel_filename.xls')
os.system('..\excel_filename.xls')
# or
sr.xlswrite_from_df('..\excel_filename2.xls', df, True)
# todo more sheets, names, small rectangles?

#%% trapezoidal integration
# see buckets.py, or

def trapz_df(df):
    index = sorted(set(df.plot_nr))
    CO2_C_mmol_m2_trapz = []
    N2O_N_mmol_m2_trapz = []
    treatments = []
    for nr in index:
        d = df[df.plot_nr == nr]
        CO2_C_mmol_m2_trapz.append(np.trapz(d.CO2_C_mmol_m2day, d.t)/86400)
        N2O_N_mmol_m2_trapz.append(np.trapz(d.N2O_N_mmol_m2day, d.t)/86400)
        treatments.append(d.treatment.values[0])
    return pd.DataFrame(index=index, data=dict(CO2_C_mmol_m2_trapz=CO2_C_mmol_m2_trapz,
                                               N2O_N_mmol_m2_trapz=N2O_N_mmol_m2_trapz,
                                               treatment=treatments))

    
print(trapz_df(df))

#Statistics see buckets.py

#%% subplots integration gothrough

#%% ginput



plt.show()
input('')

# * batch-programmer
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
