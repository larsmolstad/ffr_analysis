import numpy as np
import traceback
import requests
import bs4
import time
import pickle as pickle
import os
path = os.path.dirname(os.path.abspath(__file__))  # path of this file
path = os.path.split(os.path.split(path)[0])[0]  # grandparent folder
# these files must currently be put in the parent folder
# YR_SOUP_NAME = os.path.join(path, 'yr_beautiful_soups.pickle')
DATA_FILE_NAME = os.path.join(path, 'yr_data.pickle')


def get_yr_soup(dato):
    res = requests.get(
        'https://www.yr.no/sted/Norge/Akershus/%C3%85s/%C3%85s/almanakk.html?dato=' + dato)
    res.raise_for_status()
    return bs4.BeautifulSoup(res.text, 'lxml')


def get_all_yr_soups(start=(2015, 0o1, 0o1, 12, 0, 0, 0, 0, 0),
                     stop=None):
    t0 = time.mktime(start)
    if stop is None:
        t1 = time.time()
    else:
        t1 = time.mktime(stop)

    def t2tstr(t):
        t = time.gmtime(t)
        return "%s-%s-%s" % (t.tm_year, t.tm_mon, t.tm_mday)
    tt = np.arange(t0, t1, 86400)
    y = []
    t0 = time.time()
    for i, t in enumerate(tt):
        ts = t2tstr(t)
        print(ts, i, len(tt), (time.time() - t0) / (i + 1))
        try:
            y.append([t,  get_yr_soup(ts)])
        except:
            y.append([t, None])
            traceback.print_exc()
            print('not this one')
    return y


def parse_row(s):
    def num(s, strip=''):
        if s in [None, '-']:
            return None
        else:
            s = s.strip(strip).replace(',', '.')
            return float(s)
    v = s.text.split()
    kl = int(v[1])
    temps = [num(x, '\xb0') for x in v[3:6]]
    precip = num(v[6])
    hum = num(v[9], '%')
    return kl, temps, precip, hum


def soup2data(soup):
    tbl = soup.find_all(class_="yr-table yr-table-hourly yr-popup-area")
    q = tbl[0].find_all(scope='row')
    rows = [x.parent for x in q]
    return [parse_row(x) for x in rows]


def fix_y_times(y):
    ret = []
    for x in y:
        t = x[0]
        ty = x[1]
        for data in ty:
            kl = data[0]
            ret.append((t - 86400 / 2 + kl * 3600, data))
    return ret


def all_soups2data(y):
    return fix_y_times([(x[0], soup2data(x[1])) for x in y])


def combine_data(old_data, new_data):
    # if a day is represented in both old and new, use new
    data_dict = {d[0]: d for d in old_data}
    for d in new_data:
        data_dict[d[0]] = d
    return sorted(data_dict.values())


def save_data(data, old_data):
    pickle.dump(combine_data(data, old_data),
                open(DATA_FILE_NAME, 'wb'))


def get_stored_data():
    return pickle.load(open(DATA_FILE_NAME, 'rb'))


def make_data_file(start=(2015, 1, 1, 12, 0, 0, 0, 0, 0),
                   stop=(2015, 1, 2, 12, 0, 0, 0, 0, 0)):
    soups = get_all_yr_soups(start, stop)
    d = all_soups2data(soups)
    pickle.dump(d, open(DATA_FILE_NAME, 'wb'))


def update_weather_data():
    old_data = get_stored_data()
    last_date = max([x[0] for x in old_data])
    s = get_all_yr_soups(time.localtime(last_date - 86400))
    d = all_soups2data(s)
    updated_data = combine_data(old_data, d)
    if updated_data != old_data:
        pickle.dump(updated_data, open(DATA_FILE_NAME, 'wb'))


# def save_data_from_soup(soup):
#     pickle.dump(all_soups2data(soup), open(DATA_FILE_NAME, 'wb'))
#
#
# def get_temps(q):
#     def choose_T(T):
#         if T[0]:
#             return T[0]  # measured
#         if T[1] and T[2]:
#             return (T[1] + T[2]) / 2
#         if T[1]:
#             return T[1]
#         if T[2]:
#             return T[2]
#         return None
#     t = [x[0] for x in q]
#     T = [choose_T(x[1][1]) for x in q]
#     return t, T

# def find_day_temp(dato='2016-03-06'):
#     return soup2data(get_yr_soup(dato))
#
#
# def find_all_temps(start=(2015, 0o1, 0o1, 12, 0, 0, 0, 0, 0)):
#     t0 = time.mktime(start)
#     t1 = time.time()

#     def t2tstr(t):
#         t = time.gmtime(t)
#         return "%s-%s-%s" % (t.tm_year, t.tm_mon, t.tm_mday)
#     tt = np.arange(t0, t1, 86400)
#     y = []
#     for t in tt:
#         ts = t2tstr(t)
#         print(ts)
#         try:
#             y.append([t] + list(find_day_temp(ts)))
#         except:
#             traceback.print_exc()
#             print('not this one')
#     return y
#
#
# def parse_row_old(s):
#     def num(s, strip=''):
#         if s is None:
#             return None
#         else:
#             s = s.text.strip(strip).replace(',', '.')
#             return float(s)
#     kl = int(s.find(scope='row').text.split()[1])
#     temps = s.find_all(class_=re.compile('tempera'))
#     temps = [num(x, '\xb0') for x in temps]  # tar bort gradtegnet
#     precip = s.find(lambda x: x.text.endswith('mm'))
#     precip = num(precip, 'mm')
#     h = s.find(lambda x: x.text.endswith('%'))
#     h = num(h, '%')
#     return kl, temps, precip, h


#q = find_day_temp()
# y = get_all_yr_soups()
# pickle.dump(y, open(YR_SOUP_NAME, 'wb'))
