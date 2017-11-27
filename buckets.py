import glob
import os
import re
import divide_left_and_right
import get_data
import plot_numbering
import time
#import keyboard_interrupt_crash_workaround
import regression
import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.formula.api import ols, rlm
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from scipy.stats import norm
from plotting_compat import plt
from flux_calculations import calc_flux
import pH_data

x = np.linspace(0, 2 * np.pi)
plt.plot(np.cos(x), np.sin(x))
# from find_plot import treatments
from png_plot import plt, show
from plot_numbering import bucket_treatments
import bucket_depths
pd.set_option('display.width', 250)

def my_group(df, columns):
    s = df[columns[0]].apply(str)
    for c in columns[1:]:
        s += '_' + df[c].apply(str)
    tempname = '_'.join(columns) + str(np.random.random()).replace('.', '')
    print(tempname)
    df[tempname] = s
    groups = df.groupby(tempname)
    df.drop(tempname, axis=1, inplace=True)
    return groups


def try_parse(s):
    try:
        return get_data.parse_filename(s)
    except:
        return False


def get_old_plot_nr(name):
    I = name.find('Measure')
    if I < 0:
        I = name.find('Plot_bw')
    name = name[I:]
    return int(re.findall('\d+', name)[0])


def remove_redoings(names):
    to_remove = []
    i = len(names) - 2
    while i >= 0:
        thisnr = get_old_plot_nr(names[i])
        after = get_old_plot_nr(names[i + 1])
        if not (thisnr + 1 == after or thisnr == 24 and after == 1):
            names.pop(i)
        i -= 1


def filename_is_ok(name, xlim=[599287, 599322], ylim=[6615225, 6615250], startdate='20170828'):
    assert(len(startdate) == 8)
    parsed = try_parse(os.path.split(name)[1])
    if not parsed:
        return False
    if parsed['date'][:8] < startdate:
        return False
    p = parsed['vehicle_pos']
    x, y = p['x'], p['y']
    return (xlim[0] < x < xlim[1]) and (ylim[0] < y < ylim[1])


def regs(name, do_plot=plt, return_plot=False, subst='N2O'):
    a = get_data.get_file_data(name)
    ad = divide_left_and_right.group_all(a, cut_before=5)
    l = ad[subst]['left']
    r = ad[subst]['right']
    regl = regression.regression2(np.array(l[0]), np.array(l[1]))
    lx = np.array([l[0][0], l[0][-1]])
    ly = lx * regl.slope + regl.intercept
    regr = regression.regression2(np.array(r[0]), np.array(r[1]))
    rx = np.array([r[0][0], r[0][-1]])
    ry = rx * regr.slope + regr.intercept
    toplot = [l[0], l[1], 'r.', r[0], r[1], 'b.', lx, ly, 'r', rx, ry, 'b']
    if do_plot:
        do_plot.plot(*toplot)
    if return_plot:
        return regl.slope, regr.slope, toplot
    else:
        return regl.slope, regr.slope


def make_df(names, left=None, right=None, subst='N2O', plotter=False,
            subplots=(4, 4)):
    if left is None:
        left = []
        right = []
        for i, name in enumerate(names):
            if plotter:
                #if plotter==plt:
                #
                plotter.subplot(subplots[0], subplots[1],
                                1 + i % (subplots[0] * subplots[1]))
                #plotter.plot([1, 2, i])
                #plt.show()
                plt.cla()
                #show()
            l, r = regs(name, do_plot=plotter, subst=subst)
            #plotter.plot([1,2,2])
            left.append(l)
            right.append(r)
            if i % (subplots[0] * subplots[1]) == 0:
                print(i, end=' ')
                #plt.plot([1,2,i%13])
                if plotter == plt:
                    show()
                #plt.clf()
    sides = []
    slopes = []
    names2 = []
    for i, name in enumerate(names):
        sides.append('left')
        slopes.append(left[i])
        names2.append(name)
        sides.append('right')
        slopes.append(right[i])
        names2.append(name)
    df = pd.DataFrame({'side': sides,
                       'slope': slopes,
                       'name': names2})
    df['plot_nr'] = [get_old_plot_nr(x) for x in df.name]
    df['treatment'] = [plot_numbering.bucket_treatment(x) for x in df.plot_nr]
    parsed = [try_parse(os.path.split(x)[1]) for x in df.name]
    df['t'] = [p['t'] for p in parsed]
#    df['daynr'] = [int(x/86400) for x in df.t]
    df['day'] = [x['date'][:8] for x in parsed]
    df['gN2O_Nperdaym2'] = df.slope.apply(lambda x: calc_flux(x, 10)*0.94)*(50.0/23.5)**2*86400*14*2
    # grams per day og m2; 10 grader, kammeret tar opp ca 10%
    # bucket_diam=24; chamber_diam=50; todo sjekke
    return df


def plotnr(df, i, t0):
    l = df[df.side == 'left'][df.plot_nr == i]
    r = df[df.side == 'right'][df.plot_nr == i]
    #l = left[df.plot_nr==i]
    #r = right[df.plot_nr==i]
    t = (l.t - t0) / 86400
    plt.plot(t, l.gN2O_Nperdaym2, '.-', t, r.gN2O_Nperdaym2, 'r.-')


def plot_treatment(df, treatment, row, t0, delete_xticks=False, title=False):
    nr = sorted(set(df[df.treatment == treatment].plot_nr))
    for i, n in enumerate(nr):
        plt.subplot(6, 4, i + 1 + row * 4)
        plotnr(df, n, t0)
        plt.grid(True)
        if delete_xticks:
            plt.gca().set_xticklabels([])
        else:
            plt.gca().set_xlabel('day of year')
        if title:
            plt.gca().set_title('replicate %d' % (i + 1))


def set_ylims(lims):
    for i in range(6 * 4):
        plt.subplot(6, 4, i + 1)
        plt.gca().set_ylim(lims)


    def plot_all(df, ylims=True, t0=(2017, 1, 1, 0, 0, 0, 0, 0, 0)):
    if isinstance(t0, (list, tuple)):
        t0 = time.mktime(t0)
    plt.clf()
    tr = list(set(df.treatment))
    for i, t in enumerate(tr):
        plot_treatment(df, t, i, t0, i < len(tr) - 1, i == 0)
    if ylims:
        set_ylims([df.gN2O_Nperdaym2.min(), df.gN2O_Nperdaym2.max()])
    for i, t in enumerate(tr):
        plt.subplot(6, 4, i * 4 + 1)
        plt.gca().set_ylabel(t)
        #mp.plot('text', (min(df.t)-t0)/86400, 0.1, t)


def barplot_groups(yy, labels, plotax=plt):
    plotax = plotax.gca() #todo
    x = []
    y = []
    ticx = []
    for i, w in enumerate(yy):
        x0 = len(y) + i
        x.extend(list(range(x0, x0 + len(w))))
        y.extend(w)
        ticx.append(x0)
    plotax.cla()
    plotax.axis('auto')
    plotax.bar(x, y)
    plotax.set_xticks(ticx)
    plotax.set_xticklabels(labels, rotation=30)


def anova_means(df, which_treatments='all', column='gN2O_Nperdaym2',
                tfun=lambda x: x, means_tfun=lambda x: x,
                barplot=True, plotax=plt):
    df['transformed'] = df[column].apply(tfun)
    means = df.groupby('plot_nr').transformed.mean()
    df.drop('transformed', axis=1, inplace=True)
    trt_series = df.groupby('plot_nr').treatment.last()  # todo bedre maate
    if which_treatments == 'all':
        which_treatments = sorted(set(trt_series))
    slope_means = [means[trt_series == t].tolist() for t in which_treatments]
    slope_means = [means_tfun(x) for x in slope_means]
    anova_res = scipy.stats.f_oneway(*slope_means)
    if barplot:
        barplot_groups(slope_means, which_treatments, plotax)
        if isinstance(barplot, str):
            plt.gca().set_title(barplot)
    return anova_res, slope_means


def print_best_pvalues(ols_res):
    p = ols_res.pvalues
    c = ols_res.conf_int()
    c['p'] = p
    c = c.sort_values(by='p')
    c['x'] = [0 if x.startswith('C(day)') else x for x in c.index]
    c = c[c.x != 0].drop('x', axis=1)
    print(c[p < 0.1])


def ols_with_transform(df, f, model='y ~ C(treatment) + C(side) + C(day)', do_print=True):
    """testing ols with the transformation function f on the data in df.gN2O_Nperdaym2"""
    df['y'] = df.gN2O_Nperdaym2.apply(f)
    res = ols(model, data=df).fit()
    if do_print:
        print(res.summary())
    scipy.stats.probplot(res.resid.values, plot=plt)
    return res


def ols_transform_tests(df, ff):
    res = []
    for f in ff:
        olsres = ols_with_transform(df, f, do_print=False)
        res.append(
            {'ols': olsres, 'normaltest': scipy.stats.normaltest(olsres.resid)})
    return res


def myfun(x, x0):
    if x < x0:
        return x * 1.0 / x0 - 1 + np.log(x0)
    else:
        return np.log(x)


def make_fun(x0):
    def fun(x):
        return myfun(x, x0)
    return fun


def find_best(df, start, stop, n):
    x0 = np.linspace(start, stop, n)
    funs = [make_fun(x) for x in x0]
    res = ols_transform_tests(df, funs)
    i = np.argmax([x['normaltest'].pvalue for x in res])
    ols_transform_tests(df, [funs[i]])
    return x0[i], res[i], x0, res


def find_best2(df, start, stop, **kwargs):
    from scipy.optimize import minimize

    def fun(x):
        #        print x
        f = make_fun(x[0])
#        print ols_transform_tests(df, [f])[0]['normaltest'].pvalue
        return -np.log(ols_transform_tests(df, [f])[0]['normaltest'].pvalue)
    return minimize(fun, (start + stop) / 2, bounds=[(start, stop)], **kwargs)


def analyse(df):

    #means_anova = anova_means(df, barplot='anova of means, each treatment')
    #print "Anova of means:"
    #print means_anova

    #fit = ols('gN2O_Nperdaym2 ~ treatment', df).fit()

    #anov_table = sm.stats.anova_lm(fit, type=2)# todo sjekk type
    #print anov_table
    
    df['nr_side'] = df.plot_nr.apply(lambda x:"%02d"%x ) + '_' + df.side
    nr_side_groups = df.groupby('nr_side')
    df = df.drop('nr_side', axis=1)

    number_side_df = nr_side_groups.mean().drop('t', axis=1)
    number_side_df['treatment'] = nr_side_groups.last().treatment
    number_side_df['side'] = nr_side_groups.last().side

    df['nr_side_day'] = df.plot_nr.apply(str) + '_' + df.side + '_' + df.day
    nsd_groups = df.groupby('nr_side_day')
    df = df.drop('nr_side_day', axis=1)
    #nsd_groups = my_group(df, ('plot_nr', 'side', 'day'))
    nsd_df = nsd_groups.mean()
    nsd_df['treatment'] = nsd_groups.last().treatment
    nsd_df['side'] = nsd_groups.last().side
    nsd_df['day'] = nsd_groups.last().day
    nsd_df['soil_volume'] = nsd_groups.last().soil_volume
    # bruke merge?

    number_side_df['trapz'] = 0.0
    for key in number_side_df.index:  # x.groups.keys():
        y = nr_side_groups.get_group(key)
        trp = np.trapz(y.gN2O_Nperdaym2, y.t) / 86400
        x = number_side_df.set_value(key, 'trapz', trp)
    #barplot_means(df, means=df.groupby('plot_nr').gN2O_Nperdaym2_N_mmol_m2day.mean())

    #mod1 = 'gN2O_Nperdaym2 ~ C(treatment) + C(side)'
    #modtrapz = 'trapz ~ C(treatment) + C(side)'
    #mod2 = 'gN2O_Nperdaym2 ~ C(treatment) + C(side) + C(day)'

    #res = ols(mod1, data=df).fit()
    # print res.params

    #number_side_df_res = ols(mod1, data=number_side_df).fit()
    # print(number_side_df_res.summary())

    #number_side_trapz_res = ols(modtrapz, data=number_side_df).fit()
    # print(number_side_trapz_res.summary())
    modtrapz = 'np.log(trapz+0.005) ~ C(treatment) + C(side)'
    number_side_trapz_res = ols(modtrapz, data=number_side_df).fit()
    print(number_side_trapz_res.summary())
    #nsd_df_res = ols(mod2, data=nsd_df).fit()
    # print(nsd_df_res.summary())

    # plot residuals vs bucket depth
    # number_side_df_res.resid
    df.boxplot('gN2O_Nperdaym2', ['treatment', 'side'], rot=30)  # , ax)
    show()

    #a = find_best2(nsd_df, 0.00001, 0.9)
    #a = find_best(nsd_df, 0.000001, 0.1, 10)
    return number_side_df


names = glob.glob('c:/zip/sort_results/results/2*')
startdate = '20170823'
ok_names = [x for x in names if filename_is_ok(x, startdate=startdate)
            and re.search('Measure_[1-9]', x)]
#bw_names = [x for x in names if re.search('Plot_bw_[1-9]', x)]
# todo
# '2017-08-11-15-00-27-x599289_904642-y6615232_9609-z0_0-h-2_79697159303_both_Plot_bw_2_'
# etc
print(len(ok_names))
remove_redoings(ok_names)
# remove_redoings(bw_names)
# bw_names.pop(0)
# bw_names.pop(5)

df_all = make_df(names=ok_names, plotter=False)  # [24*8:])
df_all['soil_volume'] = [bucket_depths.soil_volumes[(df_all.ix[i].plot_nr, df_all.ix[i].side)]
                         for i in df_all.index]
#left = df_all[df_all.side == 'left']
#right = df_all[df_all.side == 'right']

day1 = ('20170823', '20170826')
day2 = ('20170830', '20170910')
day3 = ('20171013', '20171230')


def pick_days(df_all, days):
    return df_all[df_all.day >= days[0]][df_all.day <= days[1]]


df = df_all[df_all.day > '20171000']
df = pick_days(df_all, day1)
analyse(df)

plot_all(df)


def barplot_trapz(df):
    a = analyse(df)
    plt.cla()
    a.sort_index()
    treatments = sorted(a.treatment.unique())
    toplotx = []
    toploty = []
    toplot_colors = []
    ticx = []
    x = 1
    for i, tr in enumerate(treatments):
        left = a[a.treatment == tr][a.side == 'left'].trapz.values
        right = a[a.treatment == tr][a.side == 'right'].trapz.values
        toplotx.extend(list(range(x, x + len(left) + len(right))))
        ticx.append(x + 2)
        x += 2 + len(left) + len(right)
        toploty.extend(list(left) + list(right))
        toplot_colors.extend(['r'] * len(left) + ['b'] * len(right))
    plt.bar(toplotx, toploty, color=toplot_colors)
    plt.gca().set_xticks(ticx)
    plt.gca().set_xticklabels(treatments, rotation=30)
    plt.grid(True)
    plt.gca().set_ylabel('$\mathrm{g/m^2}$  maybe')


ph_df = pH_data.df
ph_df.groupby('nr').last()


def add_get_ph(df, ph_df, ph_method='CaCl2'):
    ph_df['plot_nr'] = ph_df['nr']
    tralala = ph_df.groupby('nr').last()

    def get_ph(nr):
        return tralala[tralala.plot_nr == nr][ph_method].values[0]
    df['pH'] = df.plot_nr.apply(get_ph)


def plot_ph_vs_flux(df, ph_df, days, ph_method='CaCl2'):
    df = df[df.day >= days[0]][df.day <= days[1]]
    a = analyse(df)
    add_get_ph(a, ph_df)
    tr = sorted(df.treatment.unique())
    toplot = []
    markers = '*o><^.'
    for i, t in enumerate(tr):
        toplot.append(list(a.pH[a.treatment == t].values))
        toplot.append(list(a.trapz[a.treatment == t].values))
        toplot.append(markers[i])
    plt.plot(*toplot, markersize=8)
    plt.legend(tr)
    plt.gca().set_xlabel('pH')
    plt.gca().set_ylabel('$\int_{t_0}^{t_1} \mathrm{flux\  dt}$')
    plt.grid(True)


ph_method = 'CaCl2'
plt.clf()
plt.subplot(211)
plot_ph_vs_flux(df_all, ph_df, day2, ph_method=ph_method)
plt.gca().set_xlabel('')
plt.gca().set_title('With $\mathrm{NH_4NO_3}$')
plt.subplot(212)
plot_ph_vs_flux(df_all, ph_df, day3, ph_method=ph_method)
plt.gca().set_title('With $\mathrm{NaNO_3}$')

plot_all(df_all[df_all.day>'20161025'],ylims=False)

def print_everything(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

#old
    # groups = df.groupby(['plot_nr', 'side'])
    # gdf = groups.mean()
    # gdf['treatment'] = groups.last().treatment
    # hmmm hvordan... gjor det i steden saann forelopig.

    #df['nr_side'] = df.plot_nr.apply(str) + '_' + df.side
    #nr_side_groups = df.groupby('nr_side')
    #df = df.drop('nr_side', axis=1)

    #df['nr_side_day'] = df.plot_nr.apply(str) + '_' + df.side + '_' + df.day
    #nsd_groups = df.groupby('nr_side_day')
    #df = df.drop('nr_side_day', axis=1)

    #number_side_day_means = nsd_groups.mean()
    #number_side_day_means['treatment'] = nsd_groups.last().treatment
    #number_side_day_means['side'] = nsd_groups.last().side
    #number_side_day_means['day'] = nsd_groups.last().day
    #number_side_day_means['soil_volume'] = nsd_groups.last().soil_volume
    # bruke merge?
    
    # bor kunne gjore trapz, selvomdetikkeharnoeaasii
# def test(names, do_pause=True):
#     left = [[] for i in range(24)]
#     right = [[] for i in range(24)]
#     mp.plot('fig.clf')
#     mp.subplot(111)
#     for name in names:
#         nr = get_old_plot_nr(name)
#         l, r = regs(name, do_plot=False)
#         left[nr-1].append(l)
#         right[nr-1].append(r)
#     left = np.array(left)
#     right = np.array(right)
#     if do_pause:
#         raw_input('ok')
#     mp.plot('fig.clf')
#     mp.subplot(211)
#     mp.plot(left*1000)
#     mp.subplot(212)
#     mp.plot(right*1000)
#     return left, right

# def get_plot_row_column(nr):
#     treatment = plot_numbering.bucket_treatment(nr)
#     trlist = sorted(set(plot_numbering.treatments_small_plot_numbers.values()))
#     row = trlist.index(treatment)
#     buckets_with_this_treatment = [x for x in plot_numbering.bucket_numbers
#                                    if plot_numbering.bucket_treatment(x) == 
#                                    treatment]
#     col = buckets_with_this_treatment.index(nr)
#     return row, col
        

# def test2(names, plot_on_the_way=False, texty = 0.4):
#     left = [[] for i in range(24)]
#     right = [[] for i in range(24)]
#     toplot = [[] for i in range(24)]
#     totext = [[] for i in range(24)]
#     mp.plot('fig.clf')
#     for name in names:
#         nr = get_old_plot_nr(name)
#         treatment = plot_numbering.bucket_treatment(nr)
#         row, col = get_plot_row_column(nr)
#         l, r, tp = regs(name, do_plot=plot_on_the_way, return_plot=True)
#         if plot_on_the_way:
#             mp.plot('subplot', 6, 4, row*4 + col + 1)
#             mp.plot('text', 100, texty, treatment)
#         left[nr-1].append(l)
#         right[nr-1].append(r)
#         toplot[row*4+col].extend(tp)
#         totext[row*4+col].append(treatment)
#     if not plot_on_the_way:
#         for i in range(24):
#             mp.plot('subplot', 6, 4, i+1)
#             mp.plot(*toplot[i])
#             assert(all([x == totext[i][0] for x in totext[i]]))
#             maxy = max([max(x) for x in toplot[i][1::3]])
#             mp.plot('text', 10, maxy, totext[i][0])
#     left = np.array(left)
#     right = np.array(right)


# mp.plot('fig.clf')
# mp.plot('ioff')
# test2(ok_names[:24*5])
# mp.plot('show')
# adjust_ylim(None)
# mp.plot('show')
# adjust_ylim(0.3, 0.5)
# mp.plot('ion')

# df = make_df()
# mp.plot('fig.clf')
# mp.subplot()

# barplot_means(df)
# mp.plot('fig.clf')
# mp.subplot(211)
# barplot_means(df[df.side=='left'])
# mp.subplot(212)
# barplot_means(df[df.side=='right'])
    
# dfbw = make_df(bw_names)
# mp.plot('fig.clf')
# mp.subplot(211)
# barplot_means(dfbw[dfbw.side=='left'])
# mp.plot('set_title', 'Left and right chamber driving backwards')
# mp.subplot(212)
# barplot_means(dfbw[dfbw.side=='right'])

# #df = df.drop('treatment', axis=1)
# mod = 'gN2O_Nperdaym2 ~ C(treatment) + side'
# print mod
# res = ols(mod, data=df).fit()
# ols('np.log(gN2O_Nperdaym2+0.4)  ~ C(treatment) + side', data=df).fit().summary()
# def adjust_ylim(min=0.4, max=0.5):
#     for i in range(24):
#         mp.plot('subplot',6,4,i+1)
#         if min is not None:
#             mp.plot('set_ylim', [min, max])
#         if i < 20:
#             mp.plot('set_xticklabels', [])

