"""
Functions to map raw data files to bucket numbers and to assign
the right treatments to them.

"""
import re
import os

name = 'buckets'

# for reasons of compatibility with the dict of rectangles, I am
# making a dict of functions returning True or False

def bucket_plot_identifier_fun(nr):

    def get_old_plot_nr(name):
        """ In the bucket experiment I did one run backwards. The names is in these files have the string 'Plot_bw' in them'"""
        I = name.find('Measure')
        if I < 0:
            I = name.find('Plot_bw')
        name = name[I:]
        return int(re.findall('\d+', name)[0])

    def fun(df):
        return get_old_plot_nr(df['name']) == nr

    return fun


functions = {i: bucket_plot_identifier_fun(i) for i in range(1, 25)}
rectangles = functions

treatment_names = {'N': 'Norite', 'O': 'Olivine', 'L': 'Larvikite',
                   'M': 'Marble', 'D': 'Dolomite', 'C': 'Control'}


# this is with the columnwise numbering back and forth in an S.
treatments = {i + 1: treatment_names[t]
              for i, t in enumerate('NOLOMNDCLDCOMNLDOMLCDNCM')}


known_broken = ['2017-11-03-14-19-34-x599304_71927-y6615238_67215-z0_0-h0_393826348891_both_Measure_13_']
def data_files_rough_filter(filenames):
    """filenames is a list of filenames.

    Returns a list of filenames where the files which we are sure do
    not belong to the bucket experiment have been taken away

    """
    def test(name):
        name = os.path.split(name)[1]
        broken = name in known_broken
        date_ok = name.startswith('2') and name > '2017-08-27'
        text_ok = name.find('Measure') > -1 or name.find('Plot_bw') > -1
        return date_ok and text_ok and not broken
    return [x for x in filenames if test(x)]
