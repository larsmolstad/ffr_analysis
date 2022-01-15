"""
Functions to map raw data filenames to plot numbers for the migmin experiment.
Also maps plot numbers to treatments.
"""
# todo use the small rectangles to make the migmin_field rectangles
import os

from polygon_utils_old import *
import E22


# some of the plots are not numbered the same way always in the result
# files (due to waypoint list errors).

name = 'migmin'
slopes_filename = '../migmin_slopes.txt'

def make_rectangles():

    def all_field_big_rectangles():
        # old and still used version
        large_rectangles = divide_rectangle(E22.main_rectangle, 3, 1)
        plots = []
        for r in large_rectangles:
            plots += divide_rectangle(r, 16, 1)
        return plots

    plot_indexes = [0,   1,  4,  5,  8, 11, 13, 15,
                    30, 28, 26, 25, 23, 22, 19, 18,
                    32, 35, 36, 38, 41, 43, 46, 47]
    plots = all_field_big_rectangles()
    plots_used = [plots[i] for i in plot_indexes]
    return {key + 1: x for key, x in enumerate(plots_used)}

rectangles = make_rectangles()
# my numbering was 18, 19, 22, 23, 25, 26, 28, 30,


treatment_names = {'N': 'Norite', 'O': 'Olivine', 'L': 'Larvikite',
                   'M': 'Marble', 'D': 'Dolomite', 'C': 'Control'}


# # this is with the columnwise numbering (not back and forth in an S)
# treatments = {i + 1: treatment_names[t]
#               for i, t in enumerate('NOLOMNDCDLNMOCDLOMLCDNCM')}

# this is with the columnwise numbering back and forth in an S.
# todo make clearer

treatments = {'rock_type':
              {i + 1:treatment_names[t]
                   for i, t in enumerate('NOLOMNDCLDCOMNLDOMLCDNCM')}}

treatments = {i + 1: {'rock_type':treatment_names[t]}
                for i, t in enumerate('NOLOMNDCLDCOMNLDOMLCDNCM')}

broken = []

def data_files_rough_filter(filenames, startdate, stopdate):
    """filenames is a list of filenames.

    Returns a list of filenames where the files which we are sure do
    not belong to the migmin experiment have been taken away

    """
    def is_ok(name):
        s = os.path.split(name)[1]
        date_ok = startdate <= s.replace('-','') <= stopdate
        return date_ok and name.find('_Plot_') > -1 and s not in broken
    return [x for x in filenames if is_ok(x)]

