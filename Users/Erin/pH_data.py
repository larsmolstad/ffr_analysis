import numpy as np
import pandas as pd
from plotting_compat import plt
import plot_numbering
import time
import resdir

# I have removed the data due to NMBU regulations
# todo: test, change to csv or xls, and move pH data somewhere else
data_string = open(os.path.join(resdir.raw_data_path, "../ph_data.txt")).read()

def s2data(s):
    s = s.split('\n')
    d = [x.strip() for x in s[0].split('.')]
    d = d[2] + d[1] + d[0]
    t = time.mktime(time.strptime(d, '%Y%m%d'))
    nr = []
    water = []
    CaCl2 = []
    KCl = []
    for r in s[2:]:
        rr = r.split('\t')
        nr.append(int(rr[0]))
        water.append(float(rr[1]))
        CaCl2.append(float(rr[2]))
        KCl.append(float(rr[3]))
    return pd.DataFrame({'time': t,
                         'date': d,
                         'nr': nr,
                         'water': water,
                         'CaCl2': CaCl2,
                         'KCl': KCl})  # ['date', 'nr', 'water', 'CaCl2', 'KCl']


def make_ph_df(data_string):
    d = [x.strip().replace(',', '.') for x in data_string.split('---')]
    ph_df = pd.concat([s2data(di) for di in d], ignore_index=True)
    ph_df['treatment'] = [plot_numbering.bucket_treatment(nr) for nr in ph_df.nr]
    return ph_df

ph_df = make_ph_df(data_string)

final_ph = ph_df.water[ph_df.date == max(ph_df.date)]


def plot1(df, solvent, treatment, nr, s='.-', ax=plt.plot):
    q = df[df.treatment == treatment]
    tr = plot_numbering.bucket_treatment_numbers()[treatment]
    x = q[q.nr == tr[nr]]
    t = (x.time.values - df.time.min()) / 86400
    ax.plot(t, x[solvent].values, s)


def plotsomesols(df, treatment, symb='.-', sols=['water', 'KCl', 'CaCl2']):
    for a, c in zip(sols, 'brg'):
        for i in range(4):
            plot1(df, a, treatment, i, c + symb + '-')


def show_all(df, sols=['water', 'KCl', 'CaCl2']):
    if isinstance(sols, str):
        sols = [sols]
    plt.clf()
    tr = plot_numbering.bucket_treatment_numbers()
    for i, key in enumerate(tr):
        print(key)
        plt.hold(True)
        plt.subplot(3, 2, i + 1)
        plt.grid(True)
        plt.text(0, 5.5, key)
        plt.gca().set_ylim([4.6, 7])
        plotsomesols(df, key, '.-', sols)
    plt.gca().set_xlabel('days since first pH measurement')
    plt.subplot(325)
    plt.gca().set_xlabel('days since first pH measurement')


df = ph_df[ph_df.date != '20160529']

def switch_in_df(df, nrs, column_name):
    temp = pd.options.mode.chained_assignment
    # this is to avoid the warrning about modifying a copy of dataframe.
    # Not sure if this is good
    pd.options.mode.chained_assignment = None
    a, b = nrs
    df.loc[a, column_name], df.loc[b, column_name] = df.loc[b, column_name], df.loc[a, column_name]
    pd.options.mode.chained_assignment = temp

switch_in_df(df, [183, 180], 'CaCl2')
switch_in_df(df, [207, 204], 'CaCl2')
print("do show_all(df, 'CaCl2')")
