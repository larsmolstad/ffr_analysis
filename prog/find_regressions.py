import xlwt
import pylab as plt
import read_regression_exception_list
import bisect_find
import divide_left_and_right
import get_data
import licor_indexes
import regression
import os
import sys
import time
from collections import defaultdict
import numpy as np
import pickle
sys.path.append(os.path.split(os.path.split(
    os.path.realpath(__file__))[0])[0])  # sorry

regression_errors = []


# class G:  # (G for global) # todo get rid of
#     res_file_name = 'slopes.txt'
#     directory = ''
#     interval = 100
#     co2_guides = True
#     co2_lag_time = 0
#     startdate = False
#     stopdate = False
#     filter_fun = False


def regression_quality_check_n2o(reg, side):
    try:
        reg.signal_range = reg.max_y - reg.min_y
    except:
        reg.signal_range = 0
    try:
        reg.curve_factor = abs(reg.slope)/reg.signal_range
    except:
        reg.curve_factor = -1
    if reg.signal_range > 0.14 or (side == 'left' and reg.curve_factor > 0.0081) or (side == 'right' and reg.curve_factor > 0.0066) or (reg.pval > 0.0001 and reg.pval < 0.001 and reg.signal_range > .003) or reg.curve_factor == -1 or reg.signal_range == -1:
        reg.quality_check = 'Outliers likely'
    elif reg.min_y < 0.31 or reg.min_y > 0.34 or (reg.slope < 0 and reg.max_y > 0.34):
        if reg.pval > 0.001 and reg.signal_range < 0.003:
            reg.quality_check = 'Out of range - possibly zero slope'
        elif reg.slope < 0:
            reg.quality_check = 'Out of range and negative'
        else:
            reg.quality_check = 'Out of range'
    elif reg.pval > 0.001:
        if reg.signal_range > 0.003:
            reg.quality_check = 'Fails p-test for other reason'
        else:
            reg.quality_check = 'Probably zero slope'
    else:
        reg.quality_check = ''


"""ORIGINAL VERSION
def get_regression_segments(data, regressions):
    #returns a dict of dicts of tuples of lists, res[side][substance] =
    #(t0, y0, t, y) where t0, y0 are the lists of time and measurements
    #(in seconds and ppm, so far) for the substance, and t and y are the
    #time and measurements picked for regression on given side ('left' or 'right')
    
    ar = np.array
    res = defaultdict(dict)
    for side, rdict in regressions.items():
        for substance, reg in rdict.items():
            if reg is None:
                continue
            t0, y0 = data[substance][:2]    
            #populate first two columns with all points from side **EEB I think this is just pulling all points regardless of side
            t = [ti for (I1, I2) in reg.Iswitch for ti in t0[I1:I2]]    #populate next two columns with points used in regression
            y = [yi for (I1, I2) in reg.Iswitch for yi in y0[I1:I2]]
            i0 = bisect_find.bisect_find(t, reg.start, nearest=True)
            i1 = bisect_find.bisect_find(t, reg.stop, nearest=True)
            t = t[i0:i1]
            y = y[i0:i1]
            res[side][substance] = (ar(t0), ar(y0), ar(t), ar(y))
    return res"""


def get_regression_segments(data, regressions):
    """
        returns a dict of dicts of tuples of lists, res[side][substance] =
        t_all, y_all:     All points. 
        t_side, y_side:   All points belonging to that side. ('left' or 'right')
        t_used, y_used:   All points, for that side, that were used in the regression.
        (t in seconds and y in ppm, so far) 
    """
    res = defaultdict(dict)
    for side, rdict in regressions.items():
        if side == 'filename':
            continue
        for substance, reg in rdict.items():
            if reg is None:
                continue
            # populate first two columns with all points
            t_all, y_all = data[substance][:2]
            # populate next two columns with all points from each side
            t_side = [ti for (I1, I2) in reg.Iswitch for ti in t_all[I1:I2]]
            y_side = [yi for (I1, I2) in reg.Iswitch for yi in y_all[I1:I2]]
            # populate next two columns with points used in regression
            i0 = bisect_find.bisect_find(t_side, reg.start, nearest=True)
            i1 = bisect_find.bisect_find(t_side, reg.stop, nearest=True)
            t_used = t_side[i0:i1]
            y_used = y_side[i0:i1]
            res[side][substance] = tuple(np.array(x) for x in
                                         (t_all, y_all, t_side, y_side, t_used, y_used))
    return res


def plot_regressions(regressions, data=None, normalized=True):
    """ plotting the n2o and co2 with regression lines.
    If data is None, it uses get_data.get_file_data(regressions['filename']);
    resdir.raw_data_path must be set correctly
"""
    if 'N2O' in regressions.keys():
        raise Exception("""Deprecated call to plot_regressions. The old way was 
plot_regressions(data, regressions {, normalized=True}).  The new way is
plot_regressions(regressions, {, data=None, normalized=True})""")

    if data is None:
        data = get_data.get_file_data(regressions['filename'])
    colors = {'N2O': {'left': 'r', 'right': 'g', 'between': 'b'},
              'CO2': {'left': 'k', 'right': 'c', 'between': 'y'}}

    def get_marker(side, substance):
        mark = 'o' if substance == 'N2O' and side in ['left', 'right'] else '.'
        if substance in colors and side in colors[substance]:
            return colors[substance][side] + mark
        else:
            return mark

    def get_regline(substance):
        return '-' if substance == 'N2O' else '--'

    def twoax_plot(subst, legend, t, y, marker):
        # the legend goes in ax2, so we have to do a trick: adding empty plots to
        # ax2 when plotting in ax1
        size = 8 if 'o' in marker else None
        ax[subst].plot(t, y, marker, markersize=size)
        if subst == 'N2O':
            ax2.plot([], [], marker, markersize=size)
        legends.append(legend)

    plt.clf()
    ax1 = plt.gca()
    ax1.grid()
    ax2 = ax1.twinx()
    ax = defaultdict(lambda: ax2)
    ax['N2O'] = ax1
    legends = []

    # EEB:  Is it possible this call to get_regression_segments has already been done elsewhere and can be removed?
    segments = get_regression_segments(data, regressions)

    for subst in ['N2O', 'CO2']:
        t, y = data[subst][:2]
        marker = get_marker('between', subst)
        twoax_plot(subst, subst, t, y, marker)

    for side in ['left', 'right']:
        if segments[side]:
            for subst in ['N2O', 'CO2']:
                # EEB: tside and yside are not used here. But they are left in to be compatible with new version of get_regression_segments
                t, y, tside, yside, tused, yused = segments[side][subst]
                if len(tused) == 0:
                    continue
                r = regressions[side][subst]
                marker = get_marker(side, subst)
                legend = subst + '_' + side
                twoax_plot(subst, legend, tused, yused, marker)
                yhat = tused*r.slope + r.intercept
                regline = get_regline(subst)
                twoax_plot(subst, legend, tused, yhat, marker[0] + regline)
    t, y = data['N2O'][:2]
    ax['N2O'].plot(t, y, get_marker('between', 'N2O'))

    ax2.set_xlabel('seconds')
    ax1.set_ylabel('ppm N2O')
    ax2.set_ylabel('ppm CO2')
    ax2.legend(legends)


def xls_write_raw_data_file(filename, xls_filename, data, reg, do_open=False):
    workbook = xlwt.Workbook()
    w = workbook.add_sheet('raw_data')
    # EEB column_start is leftover from when all measurements were put in same excel file
    _write_raw(filename, w,  data, reg)
    try:
        workbook.save(xls_filename)
    except IOError:
        raise IOError("You must close the old xls file")
    if do_open:
        os.startfile(xls_filename)


def _write_raw(filename, worksheet,  data, reg, column_start=0):
    #data = get_data.get_file_data(filename)
    # reg = regr.find_all_slopes(filename, do_plot=False) #EEB can we delete do_plot here?                # EEB This is also done when make images.  Combine functions?
    segments = get_regression_segments(data, reg)
    # print(segments)
    column = column_start
    w = worksheet
    # Row 0 gets filename
    w.write(0, column, filename)

    def write_columns(title, columns, column_start, under_titles):
        # Row 1 gets substance name (title)
        w.write(1, column_start, title)
        for i, vector in enumerate(columns):
            # Row 2 gets "under title" e.g. time or signal
            w.write(2, column_start, under_titles[i])
            for j, v in enumerate(vector):
                # Data starts at row 3 (which is row 4 in Excel)
                w.write(j+3, column_start, v)
            # increment so next column(s)
            column_start += 1
        return column_start
    for subst, vals in data.items():
        if not (isinstance(vals, list) and len(vals) == 2
                and isinstance(vals[0], list) and isinstance(vals[1], list)
                and len(vals[0]) == len(vals[1])):                            # Skip non-gas data items for now, e.g. filename, aux, side
            continue
        # Write time column and all measurements
        column = write_columns(subst, vals, column, ['time', 'signal'])
        t_orig, y_orig = vals
        for side in ['right', 'left']:
            if side in segments:
                if subst in segments[side]:
                    # Write all measurements attributed to each side
                    tside, yside = segments[side][subst][2:4]
                    yy = [y if t_orig[i] in tside else None
                          for i, y in enumerate(y_orig)]
                    #deb.append([segments, side, subst, t_orig])
                    column = write_columns('%s_%s_%s' % (subst, side, 'all'),
                                           [yy],  # segments[side][subst][2:],
                                           column, ['signal'])
                    # Write measurements used in each side's regression if applicable
                    tside, yside = segments[side][subst][4:6]
                    yy = [y if t_orig[i] in tside else None
                          for i, y in enumerate(y_orig)]
                    #deb.append([segments, side, subst, t_orig])
                    column = write_columns('%s_%s_%s' % (subst, side, 'used'),
                                           [yy],  # segments[side][subst][2:],
                                           column, ['signal'])
        column += 1  # Write a blank column before the next gas' columns start
    reg_attrs = ['slope', 'intercept', 'se_slope', 'se_intercept', 'mse']
    for side, regs in reg.items():
        for gas in regs.keys():
            if regs[gas] is None:
                continue
            # label columns for each regression (side_gas)
            w.write(1, column, 'reg:%s_%s' % (side, gas))
            for i, s in enumerate(reg_attrs):
                w.write(i*2+2, column, s)
                w.write(i*2+3, column, getattr(regs[gas], s))
            column += 1


def remove_zeros(t, y):
    t = np.array(t)
    y = np.array(y)
    t = t[y != 0]
    y = y[y != 0]
    return t, y


def get_filenames(directory_or_files, G):
    if isinstance(directory_or_files, (list, tuple)):
        return directory_or_files
    else:
        return get_data.select_files(directory_or_files, G)


dbg = []


class Options_manager(object):
    """Manages options for the Regressor objects. `options` are dicts on the
    form {'interval':100, 'crit': mse, 'co2_guides': True}, occasionally 
    {'interval':100, 'crit': mse, 'left':{'N2O': {'start':1, 'stop':100}}}, 
    or similar.

    Special exceptions, called specific_options, specific for each raw data
    filename, are read from an excel file and stored in
    self.specific_options_dict. These specific_options are used as far as
    possible, augmented by the default options (self.options) when the
    information in specific_options are insufficient (for example, when the
    specific_options are only concerning the left side, or only N2O.)

    """

    def __init__(self, options, specific_options_file_name=None):
        self.options = options
        self.specific_options_file_name = specific_options_file_name
        self.update_specific_options_dict()

    def update_specific_options_dict(self, exopts_filename=None):
        if exopts_filename is None:
            exopts_filename = self.specific_options_file_name
        if exopts_filename:
            self.specific_options_dict = read_regression_exception_list.parse_xls_file(
                exopts_filename)
        else:
            self.specific_options_dict = {}

    def get_specific_options(self, filename):
        if filename in self.specific_options_dict:
            return self.specific_options_dict[filename]
        else:
            return {}

    def get_options_string(self, filename):
        """Will be used to save a representation for the options used for a given
        filename (or the current one)."""
        filename = os.path.split(filename)[1]
        return repr((self.get_specific_options(filename), self.options)).replace('\t', ' ')

    def get_options(self, side, substance, filename_or_specific_options):
        if isinstance(filename_or_specific_options, str):
            filename = os.path.split(filename_or_specific_options)[1]
            specific_options = self.get_specific_options(filename)
        else:
            specific_options = filename_or_specific_options
        return self.extract_options(side, substance, specific_options, self.options)

    def enough(self, opts, substance):
        ok_key_combinations = [['slope'],
                               ['start', 'stop'],
                               ['interval', 'crit', 'CO2'],
                               ['interval', 'crit', 'co2_guides']]
        keys = list(opts) + [substance]
        for k in ok_key_combinations:
            if all([x in keys for x in k]):
                return True
        return False

    def extract_options(self, side, subst, pref, alt={}):
        """tries to get the options we need, recursively, for doing regression
         on substance subst on side side. More complicated than it should
         be, this. pref is a dict containing the preferred options, but if
         there is not information in pref, we look in alt (alternativge)

        This funciton is used when we have given specific_options for the
        regression. The specific_options may have information only for some of
        the regressions to be done (e.g., one side or one substance).  On
        the first call, pref will be the specific_options, and alt will be the
        default options

        examples:

        default = {'co2_guides':True, 'interval':10, 'crit':'heia'}
        specific_options = {'co2_guides':True, 'left': {'N2O': {'start':1, 'stop':4}}}

        extract_options('left', 'N2O', specific_options, default) => {'start': 1, 'stop': 4}

        extract_options('left', 'CO2', specific_options, default)
        => {'co2_guides': True, 'interval': 10, 'crit': 'heia'}

        """
        def add_dicts(a, b):
            return {**a, **b}

        def extract_options_popping_pref(key):
            pref2 = pref.copy()
            pref2 = pref2.pop(key)
            return self.extract_options(side, subst, pref2, pref)
        res = dict()
        if side in pref:
            res = add_dicts(extract_options_popping_pref(side), res)
        if not self.enough(res, subst):
            if subst in pref:
                res = add_dicts(extract_options_popping_pref(subst), res)
        if not self.enough(res, subst):
            res = add_dicts(pref, res)
        if not self.enough(res, subst):
            res = add_dicts(alt, res)  # var alt2
        for s in ['left', 'right', 'CO2', 'N2O']:
            res.pop(s, None)
        return res

    def __repr__(self):
        s = 'Options_manager instance with (if it is called options):\n'
        s += '   options.options = %s\n' % repr(self.options)
        s += '   options.specific_options_file_name = %s\n' % repr(
            self.specific_options_file_name)
        s += '   (options.specific_options_dict contains the specific_options)'
        return s


# [start not used yet]

class ChamberRegressions(object):
    """ Container for regressions for one chamber of robot"""

    def __init__(self):
        pass


class FileRegressions(object):
    """ Containing all the regressions for a file"""

    def __init__(self, filename):
        self.left = ChamberRegressions()
        self.right = ChamberRegressions()
        self.filename = filename
        self.rawdata_directory = ''

    def plot(self, rawdata_directory):
        self.data = get_data.get_file_data(
            os.path.join(self.rawdata_directory, self.filename))

# [end not used yet]


class Regressor(object):

    """Makes a regressor object with regression parameters given in the
        dict "options". Example:
        options = {'interval': 100, 'crit': 'steepest', 'co2_guides': True}
        regr = find_regressions.Regressor(slopes_filename, options, exception_list_filename)

        The parameter "interval" is the width in seconds of the segment
        over which to perform the regressions. The parameter "crit"
        can be 'steepest' or 'mse'; regressions will be done where the
        curves are steepest or where they have the lowest mse,
        respectively. If co2_guides==True, the interval in time where
        the co2 curve is the steepest or has the best mse is used for
        the time of regression for the N2O.

    """

    def __init__(self, slopes_file_name, options, save_options, specific_options_file_name=None, detailed_output_path=None):
        self.options = Options_manager(options, specific_options_file_name)
        self.slopes_file_name = slopes_file_name
        self.save_options = save_options
        self.plot_fun = plot_regressions
        self.do_plot = False
        self.detailed_output_path = detailed_output_path

    def _get_divide_cut_param(self, specific_options):
        def get_maybe(key, default):
            if key in specific_options:
                return specific_options[key]
            elif key in self.options.options:
                return self.options.options[key]
            else:
                return default
        return {'cut_ends': get_maybe('cut_ends', 3),
                'cut_beginnings': get_maybe('cut_beginnings', 4)}

    def find_all_slopes(self, filename_or_data, do_plot=False, given_specific_options=False):
        """Finds the regression lines for N2O and CO2, for left and right
        side
        returns {'left':{'CO2':(Regression, (x,y)),'N2O':...}, {'right': ...}}

        given_specific_options may be given; if not, specific_options will be found from the filename
        """
        if isinstance(filename_or_data, str):
            data = get_data.get_file_data(filename_or_data)
        else:
            data = filename_or_data

        if given_specific_options is False:
            specific_options = self.options.get_specific_options(
                data['filename'])
        else:
            specific_options = given_specific_options

        cut_param = self._get_divide_cut_param(specific_options)
        rawdict = divide_left_and_right.group_all(data, **cut_param)

        keys = ['CO2',  'N2O', 'CO', 'H2O', 'licor_H2O']
        regressions = {'left': {}, 'right': {}, 'filename': data['filename']}

        for side in list(regressions.keys()):
            if side == 'filename':
                continue
            tbest = None
            for key in keys:
                tbest_orig = tbest
                regressions[side][key], tbest = \
                    self._regress1(rawdict, side, key,
                                   specific_options, tbest)

                if(regressions[side][key] is not None and
                   self.options.options["correct_negatives"]):
                    specific_options, tbest = \
                        self.try_correct_negative(regressions, data, specific_options, key, side,
                                                  rawdict, tbest_orig)

            if len(list(regressions[side].keys())) == 0:
                regressions.pop(side)

        if self.do_plot or do_plot or self.save_options['save_images'] or self.save_options['show_images']:
            self.plot_fun(regressions, data=data)
            plt.pause(0.0001)

        return regressions

    def try_correct_negative(self, regressions, data, specific_options, key, side, rawdict, tbest_orig):
        # Todo I haven't tested this method since I refactored it out of find_all_slopes,
        # and have been told it is not used anymore. Probably
        # best to remove it all, with a warning
        regressions_tmp = {'left': {}, 'right': {}}
        specific_options_bcp = specific_options  # this should be deep-copied
        # print(data['filename'],specific_options_bcp)
        if((key == "N2O") and (regressions[side][key].slope < 0)):
            specific_options["co2_guides"] = False
            regressions_tmp[side][key], tbest_tmp = self._regress1(rawdict, side, key,
                                                                   specific_options, tbest_orig)
            if(regressions_tmp[side][key].slope < 0):
                specific_options = specific_options_bcp
                specific_options[side] = {'N2O': {'start': 1, 'stop': 190}}
                regressions_tmp[side][key], tbest_tmp = self._regress1(rawdict, side, key,
                                                                       specific_options, tbest_orig)
            if(regressions_tmp[side][key].slope > 0):
                regressions[side][key] = regressions_tmp[side][key]
                tbest = tbest_tmp
            else:
                specific_options = specific_options_bcp
            # this is not updating properly
            self.options.specific_options_dict[data['filename']
                                               ] = specific_options
        return specific_options, tbest

    def _regress1(self, rawdict, side, key, specific_options, tbest):
        # rawdict is like {'CO2': x1, 'N2O': x2, ...}
        # where x1 and x2 is like{'left':(t,y,Istartstop), 'right':(t,y,Istartstop)}
        # and Istartstop is [(Istart, Istop), (Istart, Istop)...]
        # (Istartstop are indexes determining
        # which part of the data has been used for the regression for one side)
        options = self.options.get_options(side, key, specific_options)
        t, y = remove_zeros(*rawdict[key][side][:2])
        if len(t) == 0:
            reg = None
        elif 'slope' in options:  # If slope was manually set
            # (EEB) This manually creates the 'reg' without calling the regression2 function.
            # It sets SE's and MSE's to zero, and start/stop to the min and max
            # (EEB) items passed to Regression:(self, intercept, slope, se_intercept, se_slope, mse, start, stop)
            reg = regression.Regression(
                t[0], options['slope'], 0, 0, 0, t[0], t[-1])
        # If user manually gave start and stop times, call regression2 via regress_within
        elif 'start' in options and 'stop' in options:
            reg = regression.regress_within(t, y, options['start'], options['stop'])
        #If the regression should use CO2's start and stop points, call regression2 via regress_within
        elif key != 'CO2' and options['co2_guides'] and tbest is not None:
            reg = regression.regress_within(t, y, *tbest)
            
        #In all other cases, call regression2 via find_best_regression, which will find the best start and stop times.
        else:
            reg = regression.find_best_regression(
                t, y, options['interval'], options['crit'])
            if key == 'CO2' and reg is not None:
                tbest = reg.start, reg.stop
        if reg is not None:
            reg.Iswitch = rawdict[key][side][2] # EEB adds switching times to reg, like [(27, 41), (67, 81), (107, 121), (147, 161), (182, 181)]
            #Quality check of N2O regressions
            if key=='N2O':
                regression_quality_check_n2o(reg, side)
        return reg, tbest

    def do_regressions(self, files, write_mode='w'):
        def print_info_maybe(i, n, t0, n_on_line):
            t = time.time()
            if t - t0 > 10:
                print('%d/%d   ' % (i+1, n), end='')
                t0 = t
                n_on_line += 1
                if n_on_line > 4:
                    n_on_line = 0
                    print('')
                sys.stdout.flush()
            return t0, n_on_line

        if not files:
            print('\nNo regressions to do\n')
            return dict(), []
        print('Doing regressions')
        n = len(files)
        t0 = time.time()
        resdict = {}
        n_on_line = 0
        errors = []
        with open(self.slopes_file_name, write_mode) as f:
            for i, name in enumerate(files):
                # do regression for each file
                t0, n_on_line = print_info_maybe(i, n, t0, n_on_line)
                try:
                    if self.save_options['show_images'] or self.save_options['save_images']:
                        plt.clf()
                    data = get_data.get_file_data(name)
                    reg = self.find_all_slopes(data)
                    self.write_result_to_file(reg, name, f)
                    resdict[os.path.split(name)[1]] = reg
                    if self.save_options['show_images'] or self.save_options['save_images']:
                        self.show_and_save_images(reg, data)
                    if self.save_options['save_detailed_excel']:
                        filename = data['filename']
                        xls_write_raw_data_file(
                            filename, 
                            os.path.join(
                                self.detailed_output_path,'Values',
                                'DetailedRawData_'+ filename+'.xls'),
                            data, reg, False)
                except Exception:
                    import traceback
                    errors.append([name, traceback.format_exc()])
                    # continue
        print_info_maybe(i, n, 0, 100000)
        print('Regression done on %d files with %d errors' %
              (len(files), len(errors)))
        if len(errors):
            regression_errors.append(errors)
            print('See find_regressions.regression_errors[-1]')
        return resdict, errors

    def show_and_save_images(self, reg, data):
        title_filename = data['filename']
        title_options = 'options: ' + self.options.get_options_string(data['filename'])
        try:
            left_QC = reg['left']['N2O'].quality_check
        except:
            left_QC = None
        try:
            right_QC = reg['right']['N2O'].quality_check
        except:
            right_QC = None
        title_left_QC = 'Left: '+left_QC if left_QC else "" 
        title_right_QC = 'Right: '+right_QC if right_QC else ""
        plt.title(title_filename + '\n' + title_options + '\n' + title_left_QC + '\n' + title_right_QC )
        if self.save_options['save_images']:
            image_name = os.path.join(self.detailed_output_path,"Images", title_filename +'.png')
            print(image_name)
            plt.savefig(image_name)
            if left_QC:
                plt.savefig(os.path.join(self.detailed_output_path, "Check", left_QC,  "LEFT " + left_QC  +" "+ title_filename +'.png'))
            elif right_QC:
                plt.savefig(os.path.join(self.detailed_output_path, "Check", right_QC, "RIGHT "+ right_QC +" "+ title_filename +'.png'))
        if self.save_options['show_images']:
            plt.show()

    def find_regressions(self, directory_or_files):
        files = get_filenames(directory_or_files, {})
        resdict, errors = self.do_regressions(files)
        with open(os.path.splitext(self.slopes_file_name)[0] + '.pickle', 'wb') as f:
            pickle.dump(resdict, f)
        return resdict
        
    def update_regressions_file(self, directory_or_files):
        """ this assumes that all files is in the same directory"""
        files = get_filenames(directory_or_files, {})
        directory = os.path.split(files[0])[0]
        lines = [x.split('\t') for x in
                 open(self.slopes_file_name, 'r').readlines()]
        if os.path.isfile(self.slopes_file_name):
            done = [(x[0], x[2]) for x in lines]  # this is name and options
        else:
            done = []
        done = [(os.path.join(directory, x[0]), x[1]) for x in done]
        must_be_done = [(x, self.options.get_options_string(x)) for x in files]
        rest = sorted(set(must_be_done) - set(done))
        q = (len(must_be_done), len(must_be_done) - len(rest))
        print('Regressions: %d files, %d already done with the same options' % q)
        files = [x[0] for x in rest]
        resdict, errors = self.do_regressions(files, 'a')
        pickle_name = os.path.splitext(self.slopes_file_name)[0] + '.pickle'
        try:
            old_dict = pickle.load(open(pickle_name), 'rb')
        except:
            print('File ', pickle_name, 'not found. Starting empty')
            old_dict = {}
        resdict = {**old_dict, **resdict}
        with open(pickle_name, 'wb') as f:
            pickle.dump(resdict, f)
    
    
    def write_result_to_file(self, res, name, f):
        options_string = self.options.get_options_string(name)
        for side, sideres in res.items():
            if side == 'filename':
                continue
            s = os.path.split(name)[1] + '\t' + side
            s += '\t' + options_string
            ok = []
            for key, regres in sideres.items():
                ok.append(regres is not None)
                if regres is None:
                    continue
                s += '\t{0}\t{1}'.format(key, regres.slope)
                s += '\t{0}_rsq\t{1}'.format(key, regres.rsq)
                s += '\t{0}_pval\t{1}'.format(key, regres.pval)
                s += '\t{0}_intercept\t{1}'.format(key, regres.intercept)
                s += '\t{0}_min\t{1}'.format(key, regres.min_y)
                s += '\t{0}_max\t{1}'.format(key, regres.max_y)
                if key=='N2O':
                    s += '\t{0}_quality_check\t{1}'.format(key, regres.quality_check)
            if all(ok):
                f.write(s + '\n')
            elif any(ok):
                print('\n******************************')
                print('In %s, %s:'%(name, side))
                print(""" 
Regressions found for at least one substance, but not all. 
This in not yet handled by this software. Line not written to file""")
                print('******************************\n')

    def __repr__(self):
        s = "Regressor with \nslopes_file_name = %s\noptions = %s"\
            % (self.slopes_file_name, repr(self.options))
        return s


def print_reg(regres):
    for k, dct in regres.items():
        if k == 'filename':
            continue
        print(k)
        for subst, reg in dct.items():
            print(subst)
            print(reg)


def make_detailed_output_folders(detailed_output_path):
    paths = [["Images"],
             ["Values"],
             ["Check", "Outliers likely"],
             ["Check", "Out of range - possibly zero slope"],
             ["Check", "Out of range and negative"],
             ["Check", "Out of range"],
             ["Check", "Fails p-test for other reason"],
             ["Check", "Probably zero slope"]]
    for p in paths:
        path = os.path.join(detailed_output_path, *p)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)


def plot_raw(filename, key='N2O'):
    "key may be 'N2O', 'CO2', 'Wind'"
    a = get_data.get_file_data(filename)
    plt.plot(a[key][0], a[key][1], '.')
    plt.gca().set_xlabel('seconds')
    if key in ['N2O', 'CO2', 'CO']:
        plt.gca().set_ylabel('ppm')
    return a

def plot_error_number(n, key='N2O'):
    name, err = regression_errors[-1][n]
    print('--------- name was: %s\nerror was:\n%s\n----------'%(name,err))
    a = plot_raw(name)
    print('shifting:', a['side'])
    return name, a
