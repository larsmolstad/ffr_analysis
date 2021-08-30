# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import pylab as plt
import xlwt 
import sys
sys.path.append(os.path.join(os.getcwd(), 'prog'))
import resdir
import get_data
import find_regressions
options = {'interval': 100, 'crit': 'steepest', 'co2_guides': True}
slopes_filename = 'justatest.txt'
specific_options_filename = 'Y:\\MINA\\Miljøvitenskap\\Jord\\FFR\\specific_options.xlsx'

regr = find_regressions.Regressor(slopes_filename, options, specific_options_filename)


"""
Created on Wed Jan 10 12:33:00 2018

"""
resdir.raw_data_path = 'Y:\\MINA\\Miljøvitenskap\\Jord\\FFR\\_RAWDATA'
#resdir.raw_data_path = '..\\..\\_RAWDATA'
filename='2017-12-12-13-55-39-x599285_624736-y6615204_70599-z0_0-h0_713586436974_both_Plot_12_' #normal
filename='2018-01-04-15-09-50-x599285_619454-y6615204_69633-z0_0-h0_719318188761_both_Plot_12_' #weird N2O slope
filename='2018-01-04-14-22-58-x599237_474308-y6615197_79174-z0_0-h-2_39559766496_left_Plot_2_' #maxed out CO2
filename='2018-01-12-11-14-35-x599311_43011-y6615319_91615-z0_0-h1_82359721564_left_ParkingTest_3_'
filename='2018-01-12-11-10-37-x599311_43011-y6615319_91615-z0_0-h1_8637217465_both_ParkingTest_2_'
filename='2018-01-12-11-06-23-x599312_724613-y6615315_4011-z0_0-h1_66725610087_right_ParkingTest_1_'

#fix single filename with directory path
if not os.path.isfile(filename):
    filename = os.path.join(resdir.raw_data_path, filename)
        

#make filenames variable filled with all of them
filenames = glob.glob(os.path.join(resdir.raw_data_path, '2*'))
#sort filenames chronologically (they are not necessarily in chronological order in folder)
filenames = sorted(filenames)

#filter files based on a string (careful!)
filenames2=[s for s in filenames if s.find('2018-01')>-1]
filenames2=[s for s in filenames if os.path.split(s)[1].startswith('2018-01')]
#filter(lambda x: x.find('2018'), filenames)
print(filenames2)

#%%
"""
simple gas chart, one gas at a time
"""

def plot_gas(filename, gas='CO2'):
    #fix single filename with directory path
    if not os.path.isfile(filename):
        filename = os.path.join(resdir.raw_data_path, filename)
    #load data for one  file
    a = get_data.get_file_data(filename)
    #make simple plot of either N2O or CO2
    plt.plot(a[gas][0], a[gas][1], '.')


"""
examples using simple gas chart
"""

#single plot
plot_gas(filename, gas='N2O')


#loop through multi plots ... the last argument is how many to skip each time
#for i in range(0, len(filenames),1000):
for i in range(18791,18875):
    print(filenames[i])
    plot_gas(filenames[i])
    plt.show()
    
#plot the last file in the list
plot_gas(filenames[-1]) 

#change the y limits
plot_gas(filename);plt.gca().set_ylim([200,800])
#OR change y limits (apparently can't use ymin or ymax alone)
plot_gas(filename);plt.gca().set_ylim(ymin=200,ymax=800)

#custom
filename='2018-01-03-13-06-39-x599263_542108-y6615221_28657-z0_0-h-2_39905751122_right_Plot_1_'
plot_gas(filename, gas='CO2');
plt.gca().set_ylim(ymin=200,ymax=800)

#%%
"""
Standard deviation of CO2
"""
for i in range(18691,18875):
    print(filenames[i])
    a = get_data.get_file_data(filenames[i])
    print(np.std(a['CO2'][1]))
    #print(a['CO2'][1])
#%%
"""
A more extensive shart showing N2O, CO2, points used, and regression lines
"""
#%%
#Display charts for all files, or a subset:  for filename in filenames[0:10]
#To run for only one file, don't highlight the for loop beginning line. will run file currently in "filename"
for filename in filenames[18645:18648]:
    filename='2018-01-04-14-17-31-x599263_5599-y6615221_32418-z0_0-h-2_40099505776_right_Plot_1_'

    if not os.path.isfile(filename):
        filename = os.path.join(resdir.raw_data_path, filename)
    reg = regr.find_all_slopes(filename, do_plot=True)
    print(filename)
    plt.savefig(os.path.join('Y:/MINA/Miljøvitenskap/Jord/FFR/images', filename+'.png'))
    print(reg)
    plt.show()
    
    #to display help just type regr
    #regr
    
#%%
"""
Export raw data to Excel
"""


def _write_raw(raw_filename, worksheet, column_start=0):
    data = get_data.get_file_data(raw_filename)
    reg = regr.find_all_slopes(raw_filename, do_plot=False)
    segments = find_regressions.get_regression_segments(data, reg)
    column = column_start
    w = worksheet
    w.write(0, column, raw_filename)
    def write_columns(title, columns, column_start, under_titles):
        w.write(1, column_start, title)
        for i, vector in enumerate(columns):
            w.write(2, column_start, under_titles[i])
            for j, v in enumerate(vector):
                w.write(j+3, column_start, v)
            column_start += 1
        return column_start
    for subst, vals in data.items():
        if not (isinstance(vals, list) and len(vals)==2\
                and isinstance(vals[0], list) and isinstance(vals[1], list)\
                and len(vals[0])==len(vals[1])):
            continue
        column = write_columns(subst, vals, column, ['time', 'signal'])
        t_orig, y_orig = vals
        for side in ['right', 'left']:  
            if side in segments:     
                if subst in segments[side]:
                    tside, yside = segments[side][subst][2:]
                    yy = [y if t_orig[i] in tside else None 
                          for i, y in enumerate(y_orig)]
                    deb.append([segments, side, subst, t_orig])
                    column = write_columns('%s_%s' % (subst, side), 
                                           [yy], #segments[side][subst][2:], 
                                           column, ['signal'])
        column += 1
    reg_attrs = ['slope', 'intercept', 'se_slope', 'se_intercept', 'mse']
    for side, regs in reg.items():
        for gas in regs.keys():
            if regs[gas] is None:
                continue
            w.write(1,column,'reg:%s_%s' % (side, gas))
            for i, s in enumerate(reg_attrs):
                w.write(i*2+2, column, s)
                w.write(i*2+3, column, getattr(regs[gas], s))
            column += 1
    return column + 2

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

xls_write_raw_data_files(filenames[-10:-4], 'test.xls', do_open=True)




#%%
"""
Excel?
"""




#%% Write to Excel.



