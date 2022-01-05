# todo the y_data.pickle file must be in granparent folder
# starting empty does not work at the moment
import numpy as np
import traceback
import requests
import bs4
import time
import pickle as pickle
import os
import sys
import json
from dateutil import parser
#import urllib
path = '/home/larsmo/test/ffr/FFR/ffr_analysis/sort_ffr_results/prog'
path = os.path.dirname(os.path.abspath(__file__))  # path of this file
path = os.path.split(os.path.split(path)[0])[0]  # grandparent folder

# this file must currently be put in the parent folder
DATA_FILE_NAME = os.path.join(path, 'yr_data.pickle')

"""Ås (NMBU) målestasjon (17850) Stasjonen ligger i Ås kommune, 92 m
o.h. Den er nærmeste offisielle målestasjon, 0,9 km fra punktet
Ås. Stasjonen ble opprettet i januar 1874. Stasjonen måler nedbør,
temperatur og snødybde. Det kan mangle data i observasjonsperioden."""

def t2tstr(t):
    t = time.gmtime(t)
    return "%s-%s-%s" % (t.tm_year, t.tm_mon, t.tm_mday)

def fix_date(dato):
    if isinstance(dato, float):
        dato = t2tstr(dato)
    d = dato.split('-')
    for i in [1,2]:
        if len(d[i]) == 1:
            d[i] = '0' + d[i]
    return '-'.join(d)

def make_url(dato, station):
    dato = fix_date(dato)
    urlstarts = \
        {'aas': 'https://www.yr.no/en/statistics/table/1-60637/Norway/Viken/%C3%85s/%C3%85s?q=',
         'samfunnet': 'https://www.yr.no/en/statistics/table/1-2246625/Norway/Viken/%C3%85s/Studentsamfunnet?q=',
         'blindern': 'https://www.yr.no/en/forecast/daily-table/1-73738/Norway/Oslo/Oslo/Blindern?q='}
    return urlstarts[station] + dato

def get_yr_soup(dato, station):
    print(make_url(dato, station))
    res = requests.get(make_url(dato, station))
    res.raise_for_status()
    return bs4.BeautifulSoup(res.text, 'lxml')

all_soups_for_debug = [0]

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
    print('fetching weather data from yr.no')
    for i, t in enumerate(tt):
        ts = t2tstr(t)
        print(ts) #print(ts, i, len(tt), (time.time() - t0) / (i + 1))
        try:
            y.append([t, get_yr_soup(ts, 'aas')])
        except:
            y.append([t, None])
            traceback.print_exc()
            print('not this one')
    all_soups_for_debug[0] = y
    return y


def all_scripts(soup):
    return soup.find_all('script')


def all_json_in_scripts(soup):
    a = all_scripts(soup)
    ret = []
    for b in a:
        #       b2 = [x for x in b.text.split('\n') if x.find("JSON.parse(")>-1]
        b2 = [x for x in str(b).split('\n') if x.find("JSON.parse(")>-1]
        for c in b2:
            try:
                start = c.find("JSON.parse(")
                start = c.find('"', start) + 1
                end = c.rfind('"')
                c = c[start:end]
                s = "**doublebacklashes**" # todoooooooooo: put back the thermometers in the chambers!
                # replacing "\\" with "\\", but not "\\\\"
                c = c.replace('\\',s).replace(s*3, "\\").replace(s,"")
                ret.append(json.loads(c))
            except:
                pass #todo
    return ret


def soup2data_new(soup):
    aj = all_json_in_scripts(soup)
    aj1 = aj[1]
    data = aj1['queries'][3]['state']['data']['historical']['days'][0]['hours']
    res = []
    for (i, d) in enumerate(data):
        try:
            t = parser.parse(d["time"]).timestamp()  #Note if verifying on yr.no: CET is UTC+1 in winter and UTC+2 in summer
            kl = i
            temp = [d["temperature"].get(s, None) for s in ["value", "max", "min"]] #Note if verifying on yr.no: python weather data is rearranged, the yr site is ordered min, max, measured value.
            precipitation = d["precipitation"].get("total", None)
            humidity = d["humidity"].get("value", None)
            res.append((t, [kl, temp, precipitation, humidity]))
        except KeyError as e:
            print("KeyError", e, time.ctime(t))
            sys.stdout.flush()
    return res


def soup2data(soup):
    try:
        return soup2data_new(soup)
    except:
        print(traceback.format_exc())
        print("soup2data_new doesn't work")
        #", trying old version")
        #return soup2data_old(soup)
        #print("That didn't work either")


def all_soups2data(time_soup_list):
    return sum([soup2data(ts[1]) for ts in time_soup_list], [])


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
    try:
        return pickle.load(open(DATA_FILE_NAME, 'rb'))
    except FileNotFoundError:
        print("weather date file %s not found, starting empty"%DATA_FILE_NAME)
        return []


def make_data_file(start=(2015, 1, 1, 12, 0, 0, 0, 0, 0),
                   stop=(2015, 1, 2, 12, 0, 0, 0, 0, 0)):
    soups = get_all_yr_soups(start, stop)
    d = all_soups2data(soups)
    pickle.dump(d, open(DATA_FILE_NAME, 'wb'))


def update_weather_data(stop=None):
    old_data = get_stored_data()
    last_date = max([x[0] for x in old_data])
    s = get_all_yr_soups(time.localtime(last_date - 86400), stop)
    d = all_soups2data(s)
    updated_data = combine_data(old_data, d)
    if updated_data != old_data:
        pickle.dump(updated_data, open(DATA_FILE_NAME, 'wb'))


def get_station_daydata(t, station):
    ts = t2tstr(t)
    print(ts)
    try:
        soup = get_yr_soup(ts, station)
        return soup2data(soup)
    except Exception as e:
        print(e)
        print('not this one: ' + ts)
        print(make_url(ts, station))
        return None
    
def update_weather_data(stop=None):
    data = get_stored_data()
    old_n = len(data)
    last_date = max([x[0] for x in data])
    t0 = last_date - 86400
    if stop is None:
        t1 = time.time()
    else:
        t1 = time.mktime(stop)
    tt = np.arange(t0, t1, 86400)
    print('fetching weather data from yr.no')
    for t in tt:
        data = get_station_daydata(t, 'aas')
        if data is None:
            data = get_station_daydata(t, 'samfunnet', t)
        if data is not None:
            data = combine_data(data, d)
        else:
            print("No data for this day: {}".format(time.ctime(t)))
    if len(data) != old_n:
        pickle.dump(data, open(DATA_FILE_NAME, 'wb'))
        

        
def my_find_all(s, substr):
    res = []
    i = s.find(substr)
    while i>=0:
        res.append(i)
        i = s.find(substr, i+1)
    return res


#--
#     q = list(aj0["statistics"]["locations"].values())[0]["days"]
#     w = list(q.values())[0]
#     e = w["data"]['historical']
# e['units']
# r = e['days'][0]['hours']
# res = {}
# res['temperature'] = [x['temperature']['value'] for x in r]
# #res['wind'] = [x['wind'] for x in r]
# res['precip'] = [x['precipitation']['total'] for x in r]
# res['humidity'] = [x['humidity']['value'] for x in r]
# t = [parser.parse(x["time"]).timestamp()  for x in r]

#Note if verifying on yr.no: CET is UTC+1 in winter and UTC+2 in summer
# update_weather_data(stop=(2020,3,10,12,0,0,0,0,0))
# update_weather_data()
# get_all_yr_soups(start=(2015, 0o1, 0o1, 12, 0, 0, 0, 0, 0), stop = (2015, 0o1, 0o7, 12, 0, 0, 0, 0, 0))
#soups = all_soups_for_debug[0]
#--


# def soup2data_old(soup):
#     aj = all_json_in_scripts(soup)
#     aj0 = aj[0]
#     q = list(aj0["statistics"]["locations"].values())[0]["days"]
#     try:
#         w = list(q.values())[0]
#         e = w["data"]['historical'] # er hele forskjellen at 'historical' er borte?
#         e['units']
#         data = e['days'][0]['hours']
#     except KeyError as e:
#         print("KeyError", e)
#         return []
#     res = []
#     for (i, d) in enumerate(data):
#         try:
#             t = parser.parse(d["time"]).timestamp()  #Note if verifying on yr.no: CET is UTC+1 in winter and UTC+2 in summer
#             kl = i
#             temp = [d["temperature"].get(s, None) for s in ["value", "max", "min"]] #Note if verifying on yr.no: python weather data is rearranged, the yr site is ordered min, max, measured value.
#             precipitation = d["precipitation"].get("total", None)
#             humidity = d["humidity"].get("value", None)
#             res.append((t, [kl, temp, precipitation, humidity]))
#         except KeyError as e:
#             print("KeyError", e, time.ctime(t))
#             sys.stdout.flush()
#     return res
