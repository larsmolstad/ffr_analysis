"""
Functions to map raw data filenames to plot numbers for the agropro experiment.
Also maps plot numbers to treatments.
"""
import os

from polygon_utils_old import *
import E22

name = 'agropro'
slopes_filename = '../agropro_slopes.txt'

s = """107	RG	D	LN
108	RG	D	HN
111	G	D	LN
112	G	D	HN
127	RG	D	LN
128	RG	D	HN
211	R	D	LN
213	G	C	LN
214	G	C	HN
227	WG	D	LN
228	WG	D	HN
305	R	D	LN
315	RG	C	LN
316	RG	C	HN
321	RG	C	LN
322	RG	C	HN
323	R	D	LN
329	RG	D	LN
330	RG	D	HN
331	RG	C	LN
332	RG	C	HN
401	R	C	LN
415	R	C	LN
423	G	D	LN
424	G	D	HN
429	R	D	LN
505	R	C	LN
507	G	D	LN
508	G	D	HN
517	G	C	LN
518	G	C	HN
521	RG	C	LN
522	RG	C	HN
527	RG	D	LN
528	RG	D	HN
605	G	C	LN
606	G	C	HN
617	R	C	LN
621	G	C	LN
622	G	C	HN
627	G 	D	LN
628	G 	D	HN"""

s = [x.split() for x in s.split('\n')]

treatments = {int(x[0]): {'mixture': x[1],
                          'rock_type': x[2],
                          'fertilizer': x[3]}
              for x in s}

# for treatment in ['rock_type', 'mixture', 'fertilizer']:
#    df[treatment] = [agropro_treatments.treatments[i][treatment] for i in df.plo

def agropro_rectangles():  
    keys = [128, 228, 127, 227, 214, 213, 112, 111, 211, 108, 107,
            332, 331, 330, 329, 429, 424, 323, 423, 322, 321, 316,
            315, 415, 305, 401, 528, 628, 527, 627, 522, 622, 521,
            621, 518, 517, 617, 508, 507, 606, 505, 605]
    small = E22.rectangles()
    return {key: small[key] for key in keys}

rectangles = agropro_rectangles()

broken = []

def data_files_rough_filter(filenames, startdate='0000', stopdate='9999'):
    """filenames is a list of filenames.

    Returns a list of filenames where the files which we are sure do
    not belong to the migmin experiment have been taken away

    """
    def is_ok(x):
        x = os.path.split(x)[1]
        return startdate < x.replace('-', '')  < stopdate and \
                x.find('_Plot_') > -1 and x not in broken
                
    return [x for x in filenames if is_ok(x)]
