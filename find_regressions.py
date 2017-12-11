import os
import sys
import time
import math
import numpy as np
import pickle
sys.path.append(os.path.split(os.path.split(
    os.path.realpath(__file__))[0])[0])  # sorry

import regression
import licor_indexes
import dlt_indexes
import get_data
import divide_left_and_right


class G:  # (G for global) # todo get rid of
    res_file_name = 'slopes.txt'
    directory = ''
    interval = 100
    co2_guides = True
    co2_lag_time = 0
    startdate = False
    stopdate = False
    filter_fun = False


def plot_regressions(data, regressions, plotfun, normalized=True, do_plot=True):
    """ plotting the n2o and co2 with regression lines. 
    The CO2 is scaled to match the N2O"""
    def min_and_max(x): return min(x), max(x)

    def stretch(y, a, b):
        return [(x + b) * a for x in y]

    def make_plotargs(key, color, a, b):
        plotargs = []
        d = data[key]

        def nrm(y):
            return stretch(y, a, b)
        plotargs.extend([d[0], nrm(d[1]), color + '.'])
        for side, rdict in regressions.items():
            reg = rdict[key]
            t0, y0 = data[key][:2]
            t = [ti for (I1, I2) in reg.Iswitch for ti in t0[I1:I2]]
            y = [yi for (I1, I2) in reg.Iswitch for yi in y0[I1:I2]]
            t = t[reg.start:reg.stop]
            y = y[reg.start:reg.stop]
            plotargs.extend([t, nrm(y), color + 'o',
                             t, nrm(reg.intercept + reg.slope * np.array(t)),
                             color])
        return plotargs
    plotargs = []
    min_n2o, max_n2o = min_and_max(data['N2O'][1])
    min_co2, max_co2 = min_and_max(data['CO2'][1])
    a = (max_n2o - min_n2o) / max(1e-10, max_co2 - min_co2)
    b = (min_n2o * max_co2 - max_n2o * min_co2) / max(1e-10, max_n2o - min_n2o)
    plotargs.extend(make_plotargs('CO2', 'b', a, b))
    plotargs.extend(make_plotargs('N2O', 'r', 1, 0))
    ymin = min_n2o
    ymax = max_n2o
    span = ymax - ymin
    if do_plot:
        plotfun(*plotargs)
    else:
        return plotargs
        #plotfun('set_ylim', ymin-span*.1, ymax+span*.1)


def remove_zeros(t, y):
    t = np.array(t)
    y = np.array(y)
    t = t[y != 0]
    y = y[y != 0]
    return t, y


def write_result_to_file(res, name, f):
    for side, sideres in res.items():
        s = os.path.split(name)[1] + '\t' + side
        for key, regres in sideres.items():
            s += '\t{0}\t{1}'.format(key, regres.slope)
        f.write(s + '\n')


def get_filenames(directory_or_files, G):
    if isinstance(directory_or_files, (list, tuple)):
         return directory_or_files
    else:
        return get_data.select_files(directory, G)


class Regressor(object):

    def __init__(self, slopes_file_name, options):
        self.options = options
        self.slopes_file_name = slopes_file_name
        

    def find_all_slopes(self, filename_or_data, plotfun=None):
        """Finds the regression lines for N2O and CO2, for left and right
        side. The parameter "interval" is the width in seconds of the
        segment over which to perform the regressions. The parameter
        "crit" can be 'steepest' or 'mse'; regressions will be done where
        the curves are steepest or where they have the lowest mse,
        respectively. If co2_guides==True, the interval in time where the
        co2 curve is the steepest or has the best mse is used for the time
        of regression for the N2O.
        
        returns {'left':{'CO2':(Regression, (x,y)),'N2O':...}, {'right': ...}}
    
        """
        if isinstance(filename_or_data, str):
            data = get_data.get_file_data(filename_or_data)
        else:
            data = filename_or_data
        resdict = divide_left_and_right.group_all(data)
        regressions = {'left': {}, 'right': {}}
        keys = ['CO2',  'N2O']
        for side in list(regressions.keys()):
            tbest = None
            for key in keys:
                t, y = remove_zeros(*resdict[key][side][:2])
                Iswitch = resdict[key][side][2]
                if key != 'CO2' and self.options['co2_guides'] and tbest is not None:
                    a = regression.regress_within(t, y, *tbest)
                else:
                    a = regression.find_best_regression(t, y, self.options['interval'], self.options['crit'])
                    if key == 'CO2' and a is not None:
                        tbest = t[a.start], t[a.stop]
                if a is not None:
                    a.Iswitch = Iswitch
                    regressions[side][key] = a
            if len(list(regressions[side].keys())) == 0:
                regressions.pop(side)
        if plotfun is not None:
            plot_regressions(data, regressions, plotfun)
        return regressions


    def do_regressions(self, files):
        n = len(files)
        t0 = time.time()
        resdict = {}
        with open(self.slopes_file_name, 'w') as f:
            for i, name in enumerate(files):
                t = time.time()
                if t - t0 > 0.5:
                    print('%d/%d' % (i, n))
                    t0 = t
                try:
                    data = get_data.get_file_data(name)
                    res = self.find_all_slopes(data)
                    write_result_to_file(res, name, f)
                    resdict[os.path.split(name)[1]] = res
                except Exception as e:
                    print(name)
                    import traceback
                    traceback.print_exc()
                    print('continuing')
                    continue
        return resdict
    
        
    def find_regressions(self, directory_or_files):
        files = get_filenames(directory_or_files, {})
        resdict = self.do_regressions(files)
        with open(os.path.splitext(self.slopes_file_name)[0] + '.pickle', 'wb') as f:
            pickle.dump(resdict, f)
    
    
    def update_regressions_file(self, directory_or_files):
        """ this assumes that all files is in the same directory"""
        files = get_filenames(directory_or_files, {})
        directory = os.path.split(files[0])
        done_files = [x.split('\t')[0] for x in open(self.slopes_file_name, 'r').readlines()]
        done_files = [os.path.join(directory, x) for x in done_files]
        files = sorted(set(files)-set(done_files))
        print(len(files))
        resdict = self.do_regressions(files)
        pickle_name = os.path.splitext(self.slopes_file_name)[0] + '.pickle'
        try:
            old_dict = pickle.load(open(pickle_name), 'rb')
        except:
            print('File ', pickle_name , 'not found. Starting empty')
            old_dict = {}
        resdict = {**old_dict, **resdict}
        with open(pickle_name, 'wb') as f:
            pickle.dump(resdict, f)

    def __repr__(self):
        return "Regressor with \nslopes_file_name=%s\noptions=%s"\
            %(self.slopes_file_name, repr(self.options)) 
    
def print_reg(regres):
    for k, dct in regres.items():
        print(k)
        for subst, reg in dct.items():
            print(subst)
            print(reg)



# def do_regressions(files, res_file_name, interval, crit, co2_guides):
#     n = len(files)
#     t0 = time.time()
#     resdict = {}
#     with open(res_file_name, 'w') as f:
#         for i, name in enumerate(files):
#             t = time.time()
#             if t - t0 > 0.5:
#                 print('%d/%d' % (i, n))
#                 t0 = t
#             try:
#                 data = get_data.get_file_data(name)
#                 # res = find_fluxes(data,interval)
#                 res = find_all_slopes(data, interval, crit, co2_guides)
#                 write_result_to_file(res, name, f)
#                 resdict[os.path.split(name)[1]] = res
#             except Exception as e:
#                 import traceback
#                 traceback.print_exc()
#                 print('continuing')
#                 continue
#     return resdict



# def find_regressions(directory_or_files,
#                      res_file_name,
#                      interval,
#                      crit='steepest',
#                      co2_guides=False,
#                      G={}):
#     files = get_filenames(directory_or_files, G)
#     resdict = do_regressions(files, res_file_name, interval, crit, co2_guides)
#     with open(os.path.splitext(res_file_name)[0] + '.pickle', 'wb') as f:
#         pickle.dump(resdict, f)


# def update_regressions_file(directory_or_files,
#                             res_file_name,
#                             interval,
#                             crit='steepest',
#                             co2_guides=False,
#                             G={}):
#     files = get_filenames(directory_or_files, G)
#     done_files = [x.split('\t')[0] for x in open(res_file_name, 'r').readlines()]
#     done_files = [os.path.join(directory, x) for x in done_files]
#     files = sorted(set(files)-set(done_files))
#     print(len(files))
#     resdict = do_regressions(files, res_file_name, interval, co2_guides)
#     pickle_name = os.path.splitext(res_file_name)[0] + '.pickle'
#     try:
#         old_dict = pickle.load(open(pickle_name), 'rb')
#     except:
#         print('File ', pickle_name , 'not found. Starting empty')
#         old_dict = {}
#     resdict = {**old_dict, **resdict}
#     with open(pickle_name, 'wb') as f:
#         pickle.dump(resdict, f)


# def find_all_slopes(filename_or_data, interval, crit='steepest', co2_guides=True,
#                     plotfun=None):
#     """Finds the regression lines for N2O and CO2, for left and right
#     side. The parameter "interval" is the width in seconds of the
#     segment over which to perform the regressions. The parameter
#     "crit" can be 'steepest' or 'mse'; regressions will be done where
#     the curves are steepest or where they have the lowest mse,
#     respectively. If co2_guides==True, the interval in time where the
#     co2 curve is the steepest or has the best mse is used for the time
#     of regression for the N2O.
    
#     returns {'left':{'CO2':(Regression, (x,y)),'N2O':...}, {'right': ...}}

#     """
#     if isinstance(filename_or_data, str):
#         data = get_data.get_file_data(filename_or_data)
#     else:
#         data = filename_or_data
#     resdict = divide_left_and_right.group_all(data)
#     regressions = {'left': {}, 'right': {}}
#     keys = 'CO2 N2O'.split()
#     for side in list(regressions.keys()):
#         tbest = None
#         for key in 'CO2 N2O'.split():
#             t, y = remove_zeros(*resdict[key][side][:2])
#             Iswitch = resdict[key][side][2]
#             if key != 'CO2' and co2_guides and tbest is not None:
#                 a = regression.regress_within(t, y, *tbest)
#             else:
#                 a = regression.find_best_regression(t, y, interval, crit)
#                 if key == 'CO2' and a is not None:
#                     tbest = t[a.start], t[a.stop]
#             if a is not None:
#                 a.Iswitch = Iswitch
#                 regressions[side][key] = a
#         if len(list(regressions[side].keys())) == 0:
#             regressions.pop(side)
#     if plotfun is not None:
#         plot_regressions(data, regressions, plotfun)
#     return regressions

