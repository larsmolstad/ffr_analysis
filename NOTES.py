#%%
import os
import glob
import pylab as plt
try:
    import my_plotter2 as mp
    plt = mp
except:
    print 'no my_plotter2, using plt'
import numpy as np

#for spyder: os.chdir('c:/zip/sort_results/sort_ffr_results/')
import resdir
import get_data
import utils
import find_regressions as fr
import plot_rectangles as pr

#%% override the default result directories:
resdir.raw_data_path = 'c:\\zip\\sort_results\\results'
resdir.slopes_path = 'c:\\zip\\sort_results'
example_file = '2016-06-16-10-19-50-x599234_725955-y6615158_31496-z0_0-h0_743558650162_both_Plot_9_'

#%% never mind this:

do_show = True
def show_and_wait():
    if do_show:
        plt.show()

# get a list of all result files
filenames = glob.glob(os.path.join(resdir.raw_data_path, '*'))
print("number of files: %d"%len(filenames))
# get the data from file number 1000
a = get_data.get_file_data(filenames[1000])
plt.cla()
plt.plot(a['N2O'][0], a['N2O'][1], '.')
# with some fluxes:
filename = os.path.join(resdir.raw_data_path, example_file)
a = get_data.get_file_data(filename)
plt.plot(a['N2O'][0], a['N2O'][1], '.')
show_and_wait()
#%% simplify working on the repl (command line)
b = utils.dict2inst(a)
print(dir(b))
# (now you can do b.N2O etc with tab completion)

#%%
plt.cla()
reg = fr.find_all_slopes(a, interval=100, co2_guides=True, plotfun=plt.plot)
fr.print_reg(reg)
show_and_wait()
# the division of the data from the two chambers is done in divide_left_and_right.py
import divide_left_and_right
ad = divide_left_and_right.group_all(a)
#%%
#plotte firkantene
plt.cla()
plt.hold(True)
rectangles = pr.migmin_field_rectangles()
pr.plot_rectangles(rectangles.values(), rectangles.keys())
plt.axis('equal')
show_and_wait()

#%%
plt.show()
raw_input('')
