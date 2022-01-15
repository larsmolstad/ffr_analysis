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
import pandas as pd
pd.options.mode.chained_assignment = None 
sys.path.append(os.path.join(os.getcwd(), 'prog'))
import pylab as plt
import resdir
import get_data
import utils
import find_regressions
import sort_results as sr
import weather_data
import flux_calculations
# import scipy.stats
# from statsmodels.stats.anova import anova_lm
# import statsmodels.api as sm
# from scipy.stats import norm
current_path = os.getcwd()


starttime = time.time()


excel_filename = "..\\..\\test.xls"

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

excel_filename_start = ""
slopes_filename = "test.txt"
slopes_filename = utils.ensure_absolute_path("../test.txt")

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

def data_files_filter(filenames, startdate='0000', stopdate='9999'):
    def is_ok(x):
        x = os.path.split(x)[1]
        return startdate < x.replace('-', '')  < stopdate
    return [x for x in filenames if is_ok(x)]

filenames = data_files_filter(all_filenames, *start_and_stopdate)      
print('number of measurement files included in this run:', len(filenames))

filenames.sort()
filenames = sorted(filenames)[6:]

regr = find_regressions.Regressor(slopes_filename, options, save_options, None, detailed_output_path)
plt.rcParams['figure.figsize'] = (15, 10) #increase the image size ... EEB this doesn't work, it follows the size previously set
#%%
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
regr.find_regressions(filenames)  #EEB: If 'show_images':True, This displays all the charts on top of each other - must be happening in regression functions
#%%
unsorted_res = sr.get_result_list_from_slope_file(slopes_filename)

df0 = sr.make_df(unsorted_res)
translations = {'N2O': 'N2O_slope', 'CO': 'CO_slope', 'CO2': 'CO2_slope', 'H2O': 'H2O_slope', 'licor_H2O': 'licor_H2O_slope'} #rename slope columns because they were just called 'CO2' or 'N2O' etc
df0.rename(columns=translations, inplace=True)
df0.sort_values('date', inplace=True)#todo flytte

#weather_data.data.update()


def finalize_df(df, precip_dt=2):
    bucket_factor = (50/23.5)**2 * 0.94
    df['Tc'] = weather_data.data.get_temp(df.t)
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

df = finalize_df(df0)
#%%
openthefineapp = True
#excel_filenames = ['..\\..\\'+excel_filename_start + '_' + s + '.xls' 
#                   for s in 'RegressionOutput slopes all_columns'.split()]

# First, the main RegressionOutput file
try:
    df.to_excel(excel_filename)
    print('Regression Output file(s) written to parent directory')
    if openthefineapp:
        os.system(excel_filenames[0])
except:
    print('Regression Output file(s) NOT written -- was it open?')
    pass

#%%


