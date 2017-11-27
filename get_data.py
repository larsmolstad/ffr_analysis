import os
import re
import math
import time
import pickle
from collections import OrderedDict

import licor_indexes
import dlt_indexes


def number_after(s, letter, decimal_symbol, start=0):
    """ Returns the floating point number and the starting and ending positios 
    of the number.
    decimal_symbol is required.
    Example:
    number_after('ab2_7cd3_14ef','d','_') => (3.14, 7, 11)"""
    def tonum(s):
        return float(s.replace('_', '.'))
    startpos = s.find(letter, start) + len(letter)
    I_decimal = s.find(decimal_symbol, startpos)
    notdig = re.search('[^\d]', s[I_decimal+1:])
    I_notdig = len(s) if notdig is None else notdig.start() + I_decimal + 1
    return tonum(s[startpos:I_notdig]), startpos, I_notdig


def parse_filename(name):
    date = name.split('x')[0][:-1]
    x = number_after(name, 'x', '_')[0]
    y = number_after(name, 'y', '_')[0]
    z = number_after(name, 'z', '_')[0]
    try: #in the beginning we did not save the heading, i think
        heading = number_after(name, '-h', '_')[0]
    except Exception:
        #print 'no heading in', name
        heading = float('nan')
    side_m = re.search('right|left|both',name)
    side = side_m.group()
    posname = name[side_m.end()+1:].strip('_')
    t = time.mktime(time.strptime(date, '%Y-%m-%d-%H-%M-%S'))
    date = date.replace('-', '')
    date = '{}-{}'.format(date[:8], date[8:])
    return {'t':t,'date':date,'name':name,
            'vehicle_pos':{'x':x,'y':y,'z':z,'side':side,'posname':posname,'heading':heading}}


def old2new(data):
    # just to convert from older format, where I stored everything in a list.
    if isinstance(data, dict):
        return data
    y = {'aux':[]}
    for x in data:
        try:
            if isinstance(x[0],str):
                y[x[0]]={'ty':x[1],'dt':x[2]['dt'],'t0':x[2]['t0']}
            else:
                raise Exception('asdf')
        except Exception as e:
            print(e)
            y['aux'].append(x)
    return y


def parse_saved_data(data, filename):
    # The data structure I save to files is different from the
    # ones I send/receive. Could change this in the future.
    # For now I made two parsing functions: parse_data and parse_saved_data
    """ converts the data as read from files to a dict with 
    substance-names as keys. Example: 
    res = parse_saved_data(cPickle.load(open(filename, 'rb')))
    print res.keys()
    t,y = res['N2O']
    """
    def sumwind(w):
        if w is None:
            return 0
        else:
            return math.sqrt(sum([x*x for x in w]))
    def pick_data(d, t0, I):
        t = [x[0] - t0 for x in d['ty']]
        y = [x[1][I] for x in d['ty']]
        return [t,y]
    def smallest_t0(data_dict):
        t0 = 1e99
        for key, v in data_dict.items():
            if isinstance(v, dict) and 't0' in v and v['t0']<t0:
                t0 = v['t0']
        return t0 if t0<1e98 else 0
    res = OrderedDict()
    t0 = smallest_t0(data)*0
    for key in data:
        if key == 'dlt':
            res['N2O'] = pick_data(data[key], t0, dlt_indexes.N2O_dry)
            res['H2O'] = pick_data(data[key], t0, dlt_indexes.H2O)
            res['CO'] = pick_data(data[key], t0, dlt_indexes.CO_dry)
        elif key == 'li-cor':
            res['CO2'] = pick_data(data[key], t0, licor_indexes.CO2)
            res['licor_H2O'] = pick_data(data[key], t0, licor_indexes.H2O)
        elif key == 'wind':
            d = data[key]['ty']
            t = [x[0] - t0 for x in d]
            res['Wind'] = [t, [sumwind(y[1]) for y in d]]
    # todo wind components
    res['aux'] = parse_filename(os.path.split(filename)[-1])
    res['side'] = data['aux']
    while '' in res['side']:res['side'].remove('')
    # todo finne ut hvorfor det er '' i data['aux']. eksempel:
    # [(1464696338.082971, 1), '', (1464696338.131722, 0), '', (1464696358.207972, 1), '', (1464696358.258012, 0), '', (1464696378.334925, 1), '', (1464696378.384402, 0), '', (1464696398.456442, 1), '', (1464696398.506836, 0), '', (1464696418.579035, 1), '', (1464696418.629181, 0), '', (1464696438.700703, 1), '', (1464696438.751395, 0), '', (1464696458.824627, 1), '', (1464696458.873863, 0), '', (1464696478.946622, 1), '', (1464696478.996109, 0), '', (1464696499.067735, 1), '', (1464696499.118345, 0), '', (1464696519.200445, 1), '', (1464696519.249015, 0), '']

    return res


def get_file_data(filename):
    with open(filename, 'rb') as f:
         a = pickle.load(f)
    return parse_saved_data(old2new(a), filename)


def selection_fun(x, G):
    y = x.startswith('20') or x.startswith('21') or x.startswith('punkt')
    y = y and os.path.splitext(x)[-1] in ['','.pickle']
    if G.startdate:
        y = y and x[:len(G.startdate)] >= G.startdate
    if G.stopdate:
        y = y and x[:len(G.stopdate)] <= G.stopdate
    if G.filter_fun:
        y = list(filter(G.filter_fun, y))
    return y


def select_files(directory,G):
    files = os.listdir(directory)
    return [os.path.join(directory, x) for x in files if selection_fun(x,G)]


def get_files_data(directory, G, write_filenames = True):
    res = []
    files = select_files(directory,G)
    n = len(files)
    for i,f in enumerate(files):
        print("%s  (%d/%d)"%(os.path.split(f)[-1], i, n))
        try:
            res.append(get_file_data(f))
        except Exception as e:
            print(e)
    return res

