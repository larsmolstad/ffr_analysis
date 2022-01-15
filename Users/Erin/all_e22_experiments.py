"""
Functions to map raw data filenames to plot numbers for the E22 small rectangles (both agropro and migmin) and migmin buckets experiment.
Also maps plot numbers to treatments.
"""
import os
import re #regex

from polygon_utils_old import *
import E22

name = 'all_e22_experiments'
slopes_filename = '../all_e22_experiments_slopes.txt'

#All small rectangles for E22. Note "dummy" plot at top to populate columns of missing plots with "_"
s = """-100	_	_	_	_
101	G	C	(HN)	migmin
102	G	C	(HN)	migmin
103	WG	A	LN	agropro
104	WG	A	HN	agropro
105	G	D	(HN)	migmin
106	G	D	(HN)	migmin
107	RG	D	LN	agropro
108	RG	D	HN	agropro
109	G	CL	(HN)	migmin
110	G	CL	(HN)	migmin
111	G	D	LN	agropro
112	G	D	HN	agropro
113	WG	C	LN	agropro
114	WG	C	HN	agropro
115	G	MM	(HN)	migmin
116	G	MM	(HN)	migmin
117	G	A	LN	agropro
118	G	A	HN	agropro
119	R	A	LN	agropro
120	R	A	HN	agropro
121	G	OL	(HN)	migmin
122	G	OL	(HN)	migmin
123	G	LR	(HN)	migmin
124	G	LR	(HN)	migmin
125	WG	A	LN	agropro
126	WG	A	HN	agropro
127	RG	D	LN	agropro
128	RG	D	HN	agropro
129	G	OL	(HN)	migmin
130	G	OL	(HN)	migmin
131	G	CL	(HN)	migmin
132	G	CL	(HN)	migmin
201	G	C	(HN)	migmin
202	G	C	(HN)	migmin
203	G	A	LN	agropro
204	G	A	HN	agropro
205	G	D	(HN)	migmin
206	G	D	(HN)	migmin
207	WG	D	LN	agropro
208	WG	D	HN	agropro
209	G	CL	(HN)	migmin
210	G	CL	(HN)	migmin
211	R	D	LN	agropro
212	R	D	HN	agropro
213	G	C	LN	agropro
214	G	C	HN	agropro
215	G	MM	(HN)	migmin
216	G	MM	(HN)	migmin
217	R	A	LN	agropro
218	R	A	HN	agropro
219	G	A	LN	agropro
220	G	A	HN	agropro
221	G	OL	(HN)	migmin
222	G	OL	(HN)	migmin
223	G	LR	(HN)	migmin
224	G	LR	(HN)	migmin
225	RG	A	LN	agropro
226	RG	A	HN	agropro
227	WG	D	LN	agropro
228	WG	D	HN	agropro
229	G	OL	(HN)	migmin
230	G	OL	(HN)	migmin
231	G	CL	(HN)	migmin
232	G	CL	(HN)	migmin
301	WG	C	LN	agropro
302	WG	C	HN	agropro
303	G	LR	(HN)	migmin
304	G	LR	(HN)	migmin
305	R	D	LN	agropro
306	R	D	HN	agropro
307	G	D	(HN)	migmin
308	G	D	(HN)	migmin
309	G	A	LN	agropro
310	G	A	HN	agropro
311	G	C	(HN)	migmin
312	G	C	(HN)	migmin
313	G	OL	(HN)	migmin
314	G	OL	(HN)	migmin
315	RG	C	LN	agropro
316	RG	C	HN	agropro
317	G	MM	(HN)	migmin
318	G	MM	(HN)	migmin
319	G	CL	(HN)	migmin
320	G	CL	(HN)	migmin
321	RG	C	LN	agropro
322	RG	C	HN	agropro
323	R	D	LN	agropro
324	R	D	HN	agropro
325	G	LR	(HN)	migmin
326	G	LR	(HN)	migmin
327	G	D	(HN)	migmin
328	G	D	(HN)	migmin
329	RG	D	LN	agropro
330	RG	D	HN	agropro
331	RG	C	LN	agropro
332	RG	C	HN	agropro
401	R	C	LN	agropro
402	R	C	HN	agropro
403	G	LR	(HN)	migmin
404	G	LR	(HN)	migmin
405	WG	D	LN	agropro
406	WG	D	HN	agropro
407	G	D	(HN)	migmin
408	G	D	(HN)	migmin
409	R	A	LN	agropro
410	R	A	HN	agropro
411	G	C	(HN)	migmin
412	G	C	(HN)	migmin
413	G	OL	(HN)	migmin
414	G	OL	(HN)	migmin
415	R	C	LN	agropro
416	R	C	HN	agropro
417	G	MM	(HN)	migmin
418	G	MM	(HN)	migmin
419	G	CL	(HN)	migmin
420	G	CL	(HN)	migmin
421	WG	C	LN	agropro
422	WG	C	HN	agropro
423	G	D	LN	agropro
424	G	D	HN	agropro
425	G	LR	(HN)	migmin
426	G	LR	(HN)	migmin
427	G	D	(HN)	migmin
428	G	D	(HN)	migmin
429	R	D	LN	agropro
430	R	D	HN	agropro
431	WG	C	LN	agropro
432	WG	C	HN	agropro
501	G	MM	(HN)	migmin
502	G	MM	(HN)	migmin
503	G	CC	(HN)	migmin
504	G	CC	(HN)	migmin
505	R	C	LN	agropro
506	R	C	HN	agropro
507	G	D	LN	agropro
508	G	D	HN	agropro
509	G	CL	(HN)	migmin
510	G	CL	(HN)	migmin
511	R	A	LN	agropro
512	R	A	HN	agropro
513	G	D	(HN)	migmin
514	G	D	(HN)	migmin
515	RG	A	LN	agropro
516	RG	A	HN	agropro
517	G	C	LN	agropro
518	G	C	HN	agropro
519	G	C	(HN)	migmin
520	G	C	(HN)	migmin
521	RG	C	LN	agropro
522	RG	C	HN	agropro
523	G	LR	(HN)	migmin
524	G	LR	(HN)	migmin
525	G	MM	(HN)	migmin
526	G	MM	(HN)	migmin
527	RG	D	LN	agropro
528	RG	D	HN	agropro
529	WG	A	LN	agropro
530	WG	A	HN	agropro
531	G	OL	(HN)	migmin
532	G	OL	(HN)	migmin
601	G	MM	(HN)	migmin
602	G	MM	(HN)	migmin
603	G	CC	(HN)	migmin
604	G	CC	(HN)	migmin
605	G	C	LN	agropro
606	G	C	HN	agropro
607	WG	D	LN	agropro
608	WG	D	HN	agropro
609	G	CL	(HN)	migmin
610	G	CL	(HN)	migmin
611	RG	A	LN	agropro
612	RG	A	HN	agropro
613	G	D	(HN)	migmin
614	G	D	(HN)	migmin
615	WG	A	LN	agropro
616	WG	A	HN	agropro
617	R	C	LN	agropro
618	R	C	HN	agropro
619	G	C	(HN)	migmin
620	G	C	(HN)	migmin
621	G	C	LN	agropro
622	G	C	HN	agropro
623	G	LR	(HN)	migmin
624	G	LR	(HN)	migmin
625	G	MM	(HN)	migmin
626	G	MM	(HN)	migmin
627	G 	D	LN	agropro
628	G 	D	HN	agropro
629	RG	A	LN	agropro
630	RG	A	HN	agropro
631	G	OL	(HN)	migmin
632	G	OL	(HN)	migmin"""

s = [x.split() for x in s.split('\n')]

treatments = {int(x[0]): {'mixture': x[1],
              'rock_type': x[2],
              'fertilizer': x[3],
              'experiment': x[4]}
for x in s}


def select_rectangles():  
    keys = [101,102,103,104,105,106,107,108,109,110,
	111,112,113,114,115,116,117,118,119,120,
	121,122,123,124,125,126,127,128,129,130,
	131,132,201,202,203,204,205,206,207,208,
	209,210,211,212,213,214,215,216,217,218,
	219,220,221,222,223,224,225,226,227,228,
	229,230,231,232,301,302,303,304,305,306,
	307,308,309,310,311,312,313,314,315,316,
	317,318,319,320,321,322,323,324,325,326,
	327,328,329,330,331,332,401,402,403,404,
	405,406,407,408,409,410,411,412,413,414,
	415,416,417,418,419,420,421,422,423,424,
	425,426,427,428,429,430,431,432,501,502,
	503,504,505,506,507,508,509,510,511,512,
	513,514,515,516,517,518,519,520,521,522,
	523,524,525,526,527,528,529,530,531,532,
	601,602,603,604,605,606,607,608,609,610,
	611,612,613,614,615,616,617,618,619,620,
	621,622,623,624,625,626,627,628,629,630,
	631,632]
    small = E22.rectangles() #calls rectangle geometry from E22.py
    return {key: small[key] for key in keys}

rectangles = select_rectangles()


broken = ['2017-11-03-14-19-34-x599304_71927-y6615238_67215-z0_0-h0_393826348891_both_Measure_13_']

def data_files_rough_filter(filenames, startdate='0000', stopdate='9999'):
    """filenames is a list of filenames.

    Returns a list of filenames where the files which we are sure do
    not belong to the migmin experiment have been taken away

    """
    def is_ok(x):
        x = os.path.split(x)[1]
        return startdate < x.replace('-', '')  < stopdate  and x not in broken
#took out:    and \ x.find('_Plot_') > -1
                
    return [x for x in filenames if is_ok(x)]

"""
The following section identifies the buckets
"""

# for reasons of compatibility with the dict of rectangles, I am
# making a dict of functions returning True or False

#NOTE bucket_plot_identifier_fun is called in find_plot, which tests if what is in "rectangles" is callable.
#If the filename contains 'Measure' or 'Plot_bw', extracts the number appearing afterward.
#If filename does NOT contain those strings, "extracts" a -99 which will fail the next test
#If extracted number is 1 through 24, function returns True, otherwise function returns False

def bucket_plot_identifier_fun(nr):

    def get_old_plot_nr(name):
        """ In the bucket experiment I did one run backwards. The names is in these files have the string 'Plot_bw' in them'"""
        I = name.find('Measure')
        if I < 0:
            I = name.find('Plot_bw')
        if I < 0:  #if it still doesn't find a known bucket plot filename
            return -99
        name = name[I:]
        return int(re.findall('\d+', name)[0]) #trim away any characters except digits

    def fun(df):
        return get_old_plot_nr(df['filename']) == nr
    return fun

#Add the bucket "rectangles" to the rectangles dictionary
#functions = {i: bucket_plot_identifier_fun(i) for i in range(1, 25)}
#rectangles.update(functions) 
rectangles.update({i: bucket_plot_identifier_fun(i) for i in range(1, 25)})

treatment_names = {'N': 'Norite', 'O': 'Olivine', 'L': 'Larvikite',
                   'M': 'Marble', 'D': 'Dolomite', 'C': 'Control'}

# this is with the columnwise numbering back and forth in an S.
bucket_treatments = {i + 1: {'rock_type':treatment_names[t],'experiment': 'buckets','mixture':'G','fertilizer':'(HN)'}
                for i, t in enumerate('NOLOMNDCLDCOMNLDOMLCDNCM')}
treatments.update(bucket_treatments)  #Add the bucket treatments to the treatments dictionary


#text_ok = name.find('Measure') > -1 or name.find('Plot_bw') > -1

