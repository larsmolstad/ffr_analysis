"""

"""
# todo use the small rectangles to make the migmin_field rectangles

from polygon_utils import *
import E22


# some of the plots are not numbered the same way always in the result
# files (due to waypoint list errors). I am numbering them simply columnwise,
# starting in north-east corner

def migmin_rectangles():
    
    def all_field_big_rectangles():
        # old and still used version
        large_rectangles = divide_rectangle(E22.main_rectangle, 3, 1)
        plots = []
        for r in large_rectangles:
            plots += divide_rectangle(r, 16, 1)
        return plots

    plot_indexes = [0, 1, 4, 5, 8, 11, 13, 15, 18, 19, 22, 23, 25, 26, 28, 30, 32,
                    35, 36, 38, 41, 43, 46, 47]
    plots = all_field_big_rectangles()
    plots_used = [plots[i] for i in plot_indexes]
    return {key + 1: x for key, x in enumerate(plots_used)}


treatment_names = {'N':'Norite', 'O':'Olivine', 'L':'Larvikite',
                       'M':'Marble', 'D':'Dolomite', 'C':'Control'}


# this is with the columnwise numbering (not back and forth in an S)
treatments = {i+1:treatment_names[t] for i,t in enumerate('NOLOMNDCDLNMOCDLOMLCDNCM')}
    

def agropro_rectangles():# todo move
    keys = [128, 228, 127, 227, 214, 213, 112, 111, 211, 108, 107,
            332, 331, 330, 329, 429, 424, 323, 423, 322, 321, 316, 315, 415, 305, 401,
            528, 628, 527, 627, 522, 622, 521, 621, 518, 517, 617, 508, 507, 606, 505, 605]
    small = small_rectangles()
    return {key: small[key] for key in keys}

# treatments = {}

# for i,t in enumerate(
#     treatments[i+1] = treatment_names[t]
