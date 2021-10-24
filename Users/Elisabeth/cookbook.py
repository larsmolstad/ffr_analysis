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
import textwrap
pd.options.mode.chained_assignment = None 
sys.path.append(os.path.join(os.getcwd(), 'prog'))
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
import xlwt 
import shutil
import errno
import re
current_path = os.getcwd()

# %% #################  EDIT THESE PARAMETERS: ################################
#

starttime = time.time()
"""Explanation""" 
# experiment:
#    The experiment file defines the rectangles and treatments to be assigned to each measurement.
#    It may also filter which measurements are read (skipping irrelevant ones).
#    To process all measurements available, use:
#        import all_e22_experiments as experiment
#    Note:  The _slopes and RegressionOutput filenames will be the name of the experiment.
#
# start_and_stopdate
#    The first and last date of the raw data that will be considered
#    Example ['20171224', '20180305']. Note that since stopdate is compared
#    to a string which includes hours, minutes and seconds, the last day '20180305'
#    will not be included unless written as e.g., '2018030524' (or just '20180306')
#
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
# remove_data_outside_rectangles:
#    True or False.
#    True:  If a measurement does not fit into any rectangle defined in the experiment file, remove the measurement.
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
#    Examples:  '..\\..\\_RAWDATA' or Y:\\Shared\\N-group\\FFR\\_RAWDATA
#
# reg_output_path:
#    Directory to save the RegressionOutput Excel file
#
# detailed_output_path:
#    Directory to save raw data details and .png images of each measurement's regression points and line
#
#######################################################################

"""Edit Parameters below""" 

import all_e22_experiments as experiment  #DO NOT CHANGE THIS
# experiment.name = 'buckets' # if using buckets
#start_and_stopdate = ['20152007', '20151213']  #YYYYMMDD 2nd date has to be one day after the last date you want
#start_and_stopdate = ['20150615', '20151111']  #YYYYMMDD 2nd date has to be one day after the last date you want
start_and_stopdate = ['20210531', '20210604']  #YYYYMMDD 2nd date has to be one day after the last date you want
redo_regressions =  True

options = {#'interval': 130, 
           'start':0,
           'stop':1850,
           'crit': 'steepest', 
           'co2_guides': False,
           'correct_negatives':False
           }

save_options= {'show_images':False,
               'save_images':True,
               'save_detailed_excel':True,
               'sort_detailed_by_experiment':True
               }

remove_redoings_time = 10 #seconds

remove_data_outside_rectangles = False


# flux_units = {'N2O': {'name': 'N2O_N_mmol_m2day', 'factor': 2 * 1000 * 86400},
#              'CO2': {'name': 'CO2_C_mmol_m2day', 'factor': 1000 * 86400}}
flux_units = {'N2O': {'name': 'N2O_N_mug_m2h', 'factor': 2 * 14 * 1e6 * 3600}, 
              'CO2': {'name': 'CO2_C_mug_m2h', 'factor': 12 * 1e6 * 3600}}


# *NOTE:  Directories must exist before the program is run.
specific_options_filename = 'Y:\\Shared\\N-group\\FFR\\specific_options.xlsx'

resdir.raw_data_path = 'Y:\\Shared\\N-group\\FFR\\_RAWDATA'
                                     
#Woops this does nothing!  You have to change the path in this file, excel_filenames
reg_output_path = 'Y:\\Shared\\N-group\\FFR'

# Can also set to False
#Woops this does nothing!  You have to change the path in find_regressions.py -> class Regressor -> self.detailed_output_path (update - Have now done this so it should work)
detailed_output_path = 'Y:\\Shared\\N-group\\FFR\\Detailed_regression_output_Unsorted'


"""Optional:  Specify an example file to see CO2 and N2O charts displayed in Python"""
example_file = '2016-06-16-10-19-50-x599234_725955-y6615158_31496-z0_0-h0_743558650162_both_Plot_9_'


# %% ################### END EDIT THESE PARAMETERS ############################

excel_filename_start = experiment.name
slopes_filename = experiment.slopes_filename
slopes_filename = utils.ensure_absolute_path(slopes_filename)

# Create Directories if they do not exist
if not os.path.exists(detailed_output_path+"\\"+"Images"):
    os.makedirs(detailed_output_path+"\\"+"Images")
if not os.path.exists(detailed_output_path+"\\"+"Values"):
    os.makedirs(detailed_output_path+"\\"+"Values")
if not os.path.exists(detailed_output_path+"\\"+"Check\\Outliers likely"):
    os.makedirs(detailed_output_path+"\\"+"Check\\Outliers likely")
if not os.path.exists(detailed_output_path+"\\"+"Check\\Out of range - possibly zero slope"):
    os.makedirs(detailed_output_path+"\\"+"Check\\Out of range - possibly zero slope")
if not os.path.exists(detailed_output_path+"\\"+"Check\\Out of range and negative"):
    os.makedirs(detailed_output_path+"\\"+"Check\\Out of range and negative")
if not os.path.exists(detailed_output_path+"\\"+"Check\\Out of range"):
    os.makedirs(detailed_output_path+"\\"+"Check\\Out of range")
if not os.path.exists(detailed_output_path+"\\"+"Check\\Fails p-test for other reason"):
    os.makedirs(detailed_output_path+"\\"+"Check\\Fails p-test for other reason")
if not os.path.exists(detailed_output_path+"\\"+"Check\\Probably zero slope"):
    os.makedirs(detailed_output_path+"\\"+"Check\\Probably zero slope")
    
# Attaches the file path to the file name
#we'll do this a lot:
def with_raw_dir(filename):
    return os.path.join(resdir.raw_data_path, filename)
def with_output_dir(filename):
    return os.path.join(reg_output_path, filename)
#    if not os.path.isfile(filename):
#        filename = os.path.join(resdir.raw_data_path, filename)

# Import data from all files selected
all_filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
print("number of measurement files from robot: %d" % len(all_filenames))
#At the moment all_e22_experiments filter only checks date. find_plot and find_treatment_names are called in make_df_from_slope_file
data_file_filter_function = experiment.data_files_rough_filter              
filenames = data_file_filter_function(all_filenames, *start_and_stopdate)      
print('number of measurement files included in this run:', len(filenames))

#Import rectangles and treatments from the whichever experiment file is being used
rectangles = experiment.rectangles
treatments = experiment.treatments
treatment_names = sr.find_treatment_names(treatments)

#FOR REFERENCE:  This is copied from all_e22_experiments.py
#treatments = {int(x[0]): {'mixture': x[1],
#              'rock_type': x[2],
#              'fertilizer': x[3],
#              'experiment': x[4]}
    

#Plot size to show within python window, unless specified otherwise later on
plt.rcParams['figure.figsize'] = (10, 6)
  
# Displays a plot of the rectangles, as specified in the experiment file (not for buckets)
plt.cla()
plot_rectangles(rectangles)
# with treatments:
plt.cla()
keys = list(rectangles)
r = [rectangles[k] for k in keys]
tr = ['_'.join(treatments[k].values()) for k in keys]# todo
plot_rectangles(r, tr)
plt.show()

# How to do regressions: The next line makes the "regressor
# object" regr which will be used further below.  It contains the
# functions and parameters for doing the regressions.  The parameters
# are collected in the dict named options. (Organizing the code this
# way makes it easier to replace the regression function with your own
# functions.) 

regr = find_regressions.Regressor(slopes_filename, options, save_options, specific_options_filename, detailed_output_path)
plt.rcParams['figure.figsize'] = (15, 10) #increase the image size ... EEB this doesn't work, it follows the size previously set

#%%""" 
"""
Define a plotting function for Regressor.  EEB:  This functionality was moved to inside find_all_slopes
"""
# e.g. give it a title, save the figures to the Y:/ drive
# EEB: Disabled this because it was called via find_regressions or update_regressions, and plotted all lines on a single image instead of separate ones.
# The legend disappeared, and other strange problems.  It also made regressions run much slower.
# Instead there is now a standalone function for saving images further down.
"""def my_plotfun(regressions, data, normalized=True):
    plt.cla()
    print(data['filename'])
    print(regressions)
    find_regressions.plot_regressions(regressions, data, normalized)
    #plt.show()
    saveimagefilename = os.path.join(images_output_path, data['filename']+'.png')
    opt = regr.options.get_options_string(data['filename'])
    plt.title(opt + '\n' + repr(regressions))
    #plt.show()
    plt.savefig(saveimagefilename)
    plt.cla()
    
regr.plot_fun = my_plotfun"""



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
#plt.clf()
#plt.subplot(1, 1, 1)

# Simple plot of 1 column from raw data
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

#%%
"""
This section is only examples of doing a single regression. 
It is not used in the main program.
"""
"""
print("\nSome examples:")



# Get the data from file number 1000 (or the last one if there are
# less than 1000 files)
n = 1000 if len(all_filenames)>1000 else len(all_filenames)
# Plot the raw N2O points from file #1000 (or last file in directory)
plt.cla()
### a = plot_raw(all_filenames[n])
###plt.show()

# Get the data from example_file (set in options at the top)
examplefilename = with_raw_dir(example_file)
# checking that it exists first to avoid that this script stops:
if os.path.isfile(examplefilename):
    plt.cla()
###    plot_raw(examplefilename, 'N2O')
###    plt.show()
else:
    print('skipping example file')
    print('(does %s exist?)' % examplefilename)

# also,
# a = get_data.get_file_data(filename)
# plt.plot(a['N2O'][0], a['N2O'][1], '.')

# Do a single regression:
plt.cla()
###reg = regr.find_all_slopes(examplefilename, do_plot=True)
###plt.show()
# Print the slopes, intercepts, mse etc:
###find_regressions.print_reg(reg)
# another way:
# data = get_data.get_file_data(filename)
# reg = regr.find_all_slope(data)
"""

# %% Do many regressions
"""
This is the main program, which does regressions on all measurements selected.
This may take a long time if many measurements are selected, and redo_regressions = True.
"""

if redo_regressions:
    regr.find_regressions(filenames)  #EEB: If 'show_images':True, This displays all the charts on top of each other - must be happening in regression functions
else:
    # update resfile without redoing regressions:
    regr.update_regressions_file(filenames)

#Display error message if regression errors.  EEB: Is this ever called? Or the user has to call it?
def plot_error_number(n, key='N2O'):
    name, err = find_regressions.regression_errors[-1][n]
    print('--------- name was: %s\nerror was:\n%s\n----------'%(name,err))
    a = plot_raw(name)
    print('shifting:', a['side'])
    return name, a

#%%
"""
Preparing the data for "RegressionOutput" Excel export
"""
# %% Sort results according to the rectangles, put them in a Pandas dataframe
pd.set_option('display.width', 120)
# The slopes have been stored in the file whose name equals the value of
# slope_filename. make_df_from_slope_file picks the ones that are inside
# rectangles
# EEB This displays how many lines are read in from the slopes file, and how many after remove duplicates

df, df0 = sr.make_df_from_slope_file(slopes_filename,
                                     rectangles,
                                     treatments,
                                     remove_redoings_time, #seconds
                                     remove_data_outside_rectangles)

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

# Adds temperature from external weather data, calculates flux
# Flux unit is mol/(sec*m2), followed by other units specified in options
# calls flux_calculations.calc_flux()
"""def finalize_df_old(df, precip_dt=2):
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    # df['precip2'] = weather_data.data.get_precip2(df.t, [0, precip_dt]) #todo failed
    N2O_mol_secm2 = flux_calculations.calc_flux(df.N2O_slope, df.Tc)
    CO2_C_mol_secm2 = flux_calculations.calc_flux(df.CO2_slope, df.Tc)
    
    if experiment.name == 'all_e22_experiments':
        for i in range(len(df)):
            if df['experiment'].iloc[i] == 'buckets':
                N2O_mol_secm2.iloc[i] = N2O_mol_secm2.iloc[i]* (50/23.5)**2 * 0.94
#    if experiment.name == 'all_e22_experiments':
#       df['experiment'].apply(lambda x: 1 if x == 'buckets' else 0)
#           
    if experiment.name == 'buckets':
        N2O_mol_secm2 = N2O_mol_secm2 * (50/23.5)**2 * 0.94 # ** means to the power of
        CO2_C_mol_secm2 = CO2_C_mol_secm2 * (50/23.5)**2 *0.94
    df['N2O_mol_m2s'] = N2O_mol_secm2
    df['CO2_mol_m2s'] = CO2_C_mol_secm2
    Nunits = flux_units['N2O']
    Cunits = flux_units['CO2']
    df[Nunits['name']] = Nunits['factor'] *  N2O_mol_secm2
    df[Cunits['name']] = Cunits['factor'] *  CO2_C_mol_secm2
    df = sr.rearrange_df(df)
    return df"""

def finalize_df(df, precip_dt=2):
    bucket_factor = (50/23.5)**2 * 0.94
    df['Tc'] = weather_data.data.get_temp(df.t)
    df['precip'] = weather_data.data.get_precip(df.t)
    df['N2O_mol_m2s'] = flux_calculations.calc_flux(df.N2O_slope, df.Tc)
    df['CO2_mol_m2s'] = flux_calculations.calc_flux(df.CO2_slope, df.Tc)
    if experiment.name == 'buckets':
        df.N2O_mol_m2s *= bucket_factor
        df.CO2_mol_m2s *= bucket_factor
    elif 'experiment' in df.columns:
        # see https://www.dataquest.io/blog/settingwithcopywarning/
        df.loc[df.experiment=='buckets', 'N2O_mol_m2s'] *= bucket_factor
        df.loc[df.experiment=='buckets', 'CO2_mol_m2s'] *= bucket_factor  
    Nunits = flux_units['N2O']
    Cunits = flux_units['CO2']
    df[Nunits['name']] = Nunits['factor'] *  df.N2O_mol_m2s
    df[Cunits['name']] = Cunits['factor'] *  df.CO2_mol_m2s
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



#%%
"""
Writing the regression output data to an Excel file
"""
# %% Write Regression Output to Excel.
# "..\filename.xls" makes filename.xls in the parent directory (..\)
# experiment_RegressionOutput.xls:  The regressions and all relevant information about that measurement, including the regression options
# experiment_slopes.xls and experiment_all_columns.xls:  The columns sorted by date 
#(Not sure how to use _slopes or _all_columns ... same info is in _RegressionOutput)

openthefineapp = False
excel_filenames = ['..\\..\\'+excel_filename_start + '_' + s + '.xls' 
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

towrite = ['N2O_slope', 'CO2_slope', 'filename']
try:
    sr.xlswrite_from_df(excel_filenames[1], df, openthefineapp, towrite) #make tabs only for df.columns specified in 'towrite'
    # or if you want it all:
    #sr.xlswrite_from_df(excel_filenames[2], df, openthefineapp, df.columns) #make separate tab for each item in df.columns)
except:
    pass
# todo more sheets, names, small rectangles?
    


#%%
"""
Move Detailed_regression_output files into their own folders based on experiment.
NOTE:   It was not possible to save these files in the proper folder at the time they are created (in do_regressions)
        Because the "experiment" is only determined later, here in the cookbook.
        do_regressions writes all the Detailed_regression_output files, and also writes the "slopes" text file.
        Later, the cookbook reads the "slopes" text file and puts it in a pandas datafram, for writing to Excel "xxx_RegressionOutput.xls"
        And in the process it looks up 'experiment' (based on for example all_e22_experiments.py), and the weather, and attaches them as new columns.
"""
if save_options['sort_detailed_by_experiment']==True:
    print("Please wait while files are moved into experiment folders, this may take some time. You can turn this off by changing sort_detailed_by_experiment to False. \n\n...")

    for index, row in df.iterrows():
        print(row['filename'])
        serverfilenames = glob.iglob(detailed_output_path+"\**\*", recursive=True) #Annoyingly, this has to be done every time because the glob empties out after the loop. But this is fast. It's the string comparisons that are slow.
        #print(index)#.iterrows():  
        for serverfilename in serverfilenames: 
            #print(serverfilename)
            if row['filename'][0:19] in serverfilename: #Save time by only comparing date string within filename (first 19 characters)
                #print("MATCH "+row['filename'])
                if row['experiment']!='_': #skip any measurement that doesn't belong to a defined experiment
                    newserverfilename = serverfilename.replace("Detailed_regression_output_Unsorted","Detailed_regression_output_"+row['experiment'])
                    #newserverfilename = os.path.join(os.path.split(serverfilename)[0]+'\\'+row['experiment']+' '+os.path.split(serverfilename)[1])
                    #print(newserverfilename)
                    try:
                        shutil.move(serverfilename,newserverfilename)
                    except OSError as e:
                        #print("exception reached")
                        if e.errno != errno.ENOENT:   #Requires some library to use errno functionality
                            raise
                        # try creating parent directories
                        try:
                            os.makedirs(os.path.dirname(newserverfilename))
                            #print("trying to make dir")
                        except:
                            #print("failed make dir")
                            pass
                        shutil.move(serverfilename,newserverfilename)

"""
Copy image files which have not been caught by any checks to their own folder

imagefilenames = glob.iglob("Y:\\MINA\\Miljøvitenskap\\Jord\\FFR\\Erin\\20190620 overwinter using whole range with Checks (should be identical)\\Images\\*", recursive=True)
for imagefilename in imagefilenames: 
    try:
        #extract date from filename string. it will fail on files that aren't measurements, so put in "try"
        imagefiledatesubstring = (re.search('\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', imagefilename).group(0))
    except:
        pass
    copyimagefile = True
    #get list of files already caught by a check
    checkedfilenames = glob.iglob("Y:\\MINA\\Miljøvitenskap\\Jord\\FFR\\Erin\\20190620 overwinter using whole range with Checks (should be identical)\\Check\\*\\*", recursive=True)
    for checkedfilename in checkedfilenames: 
        if imagefiledatesubstring in checkedfilename:
            copyimagefile = False
    if copyimagefile == True:    
        newimagefilename = imagefilename.replace("Images","ImagesExcludingCheck")
        try:
            shutil.copy(imagefilename,newimagefilename)
        except OSError as e:
            if e.errno != errno.ENOENT:   #Requires some library to use errno functionality
                raise
            # try creating parent directories
            try:
                os.makedirs(os.path.dirname(newimagefilename))
            except:
                pass
            shutil.copy(imagefilename,newimagefilename)

"""    



"""
            print ("\nRUNFILE:"+row['filename'])
            print("SERVER FILES:")
            print (serverfile)
            print(df['filename'])
            
for runfile in df['filename']:
    if index==1:
        print(row['filename'])
        print(row['experiment'])    
        
        
for i in df['filename']:
    filestomove=find()
    print(i)
    
for file_name in os.listdir(detailed_output_path/*):
     if fnmatch.fnmatch(file_name, '*.txt'):
         print(file_name)
"""


#%%
"""
Old version of writing image files... this was moved to find_regressions
We originally wrote a save-image plotting function for the Regressor class, but it did not work - see notes at:  def my_plotfun
"""
#Create image files of data points and regressions
#for filename in filenames if images_output_path else []:
#    #print(filename)
#    #print(os.path.split(filename)[1])
#    title1 = os.path.split(filename)[1]
#    title2 = 'options: ' + regr.options.get_options_string(filename)
#    #title3 = repr(reg)  #This doesn't work because find_all_slopes has not been run yet, and title must be set before plot is made
#    plt.title(title1 + '\n' + title2)
#    reg = regr.find_all_slopes(filename, do_plot=True)
#    plt.savefig(os.path.join(images_output_path, os.path.split(filename)[1] +'.png'))
#    plt.clf()
    #plt.show()



#%%
"""
Old version of create detailed raw data excel ... this was moved to find_regressions
Create RAW data file with individual data points from measurement, 
indicating which points were used in the regression, and the regression values.
"""

# Arrange detailed raw data for one measurement into columns for export 
#def _write_raw(filename, worksheet, column_start=0):
#    data = get_data.get_file_data(filename)                             
#    reg = regr.find_all_slopes(filename, do_plot=False) #EEB can we delete do_plot here?                # EEB This is also done when make images.  Combine functions?
#    segments = get_regression_segments_excel(data, reg)
#    column = column_start
#    w = worksheet
#    w.write(0, column, filename)
#    def write_columns(title, columns, column_start, under_titles):
#        w.write(1, column_start, title)
#        for i, vector in enumerate(columns):
#            w.write(2, column_start, under_titles[i])
#            for j, v in enumerate(vector):
#                w.write(j+3, column_start, v)
#            column_start += 1
#        return column_start
#    for subst, vals in data.items():
#        if not (isinstance(vals, list) and len(vals)==2\
#                and isinstance(vals[0], list) and isinstance(vals[1], list)\
#                and len(vals[0])==len(vals[1])):                            # Skip non-gas data items for now, e.g. filename, aux, side
#            continue
#        column = write_columns(subst, vals, column, ['time', 'signal'])     #Write time column and all measurements
#        t_orig, y_orig = vals
#        for side in ['right', 'left']:  
#            if side in segments:     
#                if subst in segments[side]:
#                    tside, yside = segments[side][subst][0:2]               #Write all measurements attributed to each side
#                    yy = [y if t_orig[i] in tside else None 
#                          for i, y in enumerate(y_orig)]
#                    #deb.append([segments, side, subst, t_orig])
#                    column = write_columns('%s_%s_%s' % (subst, side, 'all'),         
#                                           [yy], #segments[side][subst][2:], 
#                                           column, ['signal'])
#                    tside, yside = segments[side][subst][2:]                #Write measurements used in each side's regression if applicable
#                    yy = [y if t_orig[i] in tside else None 
#                          for i, y in enumerate(y_orig)]
#                    #deb.append([segments, side, subst, t_orig])
#                    column = write_columns('%s_%s_%s' % (subst, side, 'used'),         
#                                           [yy], #segments[side][subst][2:], 
#                                           column, ['signal'])
#        column += 1                                                         #Write a blank column before the next gas' columns start
#    reg_attrs = ['slope', 'intercept', 'se_slope', 'se_intercept', 'mse']
#    for side, regs in reg.items():
#        for gas in regs.keys():
#            if regs[gas] is None:
#                continue
#            w.write(1,column,'reg:%s_%s' % (side, gas))                     #label columns for each regression (side_gas)
#            for i, s in enumerate(reg_attrs):
#                w.write(i*2+2, column, s)
#                w.write(i*2+3, column, getattr(regs[gas], s))
#            column += 1
#    return column + 2
#
## Write detailed raw data into Excel 
#def xls_write_raw_data_file(filename, xls_filename, 
#                                column_start=0, do_open=False):
#    workbook = xlwt.Workbook()
#    w = workbook.add_sheet('raw_data')
#    column_start =_write_raw(filename, w, column_start)  #EEB column_start is leftover from when all measurements were put in same excel file
#    try:
#        workbook.save(xls_filename)
#    except IOError:
#        raise IOError("You must close the old xls file")
#    if do_open:
#        os.startfile(xls_filename)
#
## Call xls_write_raw_data_file, which calls _write_raw.  Passes detailed_output_path and specifies do_open. (Don't recommend do_open for many files!)
#for filename in filenames if detailed_output_path else []:
#    xls_write_raw_data_file(filename, os.path.join(detailed_output_path+'\\Values','DetailedRawData_'
#                                                      +os.path.split(filename)[1]
#                                                      +'.xls'), do_open=False)

"""
# Original version of xls_write_raw_data_files:  
# All measurements were in one file but there was a limit of 256 columns, so only 7 or 8 measurements could fit.
def xls_write_raw_data_files(raw_filenames, xls_filename, 
                                column_start=0, do_open=False):
    assert(isinstance(raw_filenames, list))
    workbook = xlwt.Workbook()
    w = workbook.add_sheet('raw_data')
    for name in raw_filenames:
        column_start =_write_raw(name, w, column_start)
    try:
        workbook.save(xls_filename)
    except IOError:
        raise IOError("You must close the old xls file")
    if do_open:
        os.startfile(xls_filename)
    print('Wrote raw data to Excel.')

#Original call to function xls_write_raw_data_files 
#xls_write_raw_data_files(filenames[1:3], with_output_dir(excel_filename_start+'_RawDataDetails.xls'), do_open=False)
"""
#%%
"""
Further Analysis (These were developed with MIGMIN and the buckets in mind)
"""
# %%barmaps

# _ = sr.barmap_splitted(df, theta=0)

# %% trapezoidal integration to calculate the emitted N2O over a period of time:

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

# %% Just the days with high fluxes:

df2 = sr.filter_for_average_slope_days(df, 0.0005)

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


print('\n\nBarplot of trapezoidal integrations from %s to %s:'%
      (df.date.min(), df.date.max()))
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
elif experiment.name in ['all_e22_experiments']:
    treatments = ['mixture', 'fertilizer', 'rock_type', 'experiment']
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

print('run time was', int(time.time()-starttime), 'seconds')

