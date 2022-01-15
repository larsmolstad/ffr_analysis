import os
import requests
import pandas as pd
import pickle
import bisect
# import datetime
import time
import calendar
# There is also soil termperature, radiation, etc. Do:
# $ curl -X GET --user c5afa9d7-d06b-41a3-9ae4-5a5418b6c792: https://frost.met.no/observations/availableTimeSeries/v0.jsonld?sources=SN17850&referencetime=2017-01-01
# to see what data can be found

path = os.path.dirname(os.path.abspath(__file__))  # path of this file
path = os.path.split(path)[0]  # parent folder
#path = '/home/larsmo/div/ffr/merge/ffr_analysis/'
# this file must currently be put in the parent folder
DATA_FILE_NAME = os.path.join(path, 'metno_data.pickle')

try:
    id_file = os.path.join(path, 'metno_client_id.txt')
    client_id = [open(id_file).readlines()[0].strip()]
except FileNotFoundError:
    print(id_file + ' not found')
    client_id = [None]

endpoint = 'https://frost.met.no/observations/v0.jsonld'

def set_client_id(s):
    client_id[0] = s

def str2epocht(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ"))

def get_ty_from_json(json):
    data = json['data']
    res = []
    for x in data:
        t  = round(str2epocht(x['referenceTime']))
        a = dict()
        for ob in x['observations']:
            a[ob['elementId']] = ob['value']
        res.append((t, a))
    return res

def t2tstr(t):
    t = time.localtime(t) # not time.gmtime?
    return "%s-%s-%s" % (t.tm_year, t.tm_mon, t.tm_mday)

def fix_date(dato):
    if isinstance(dato, float):
        dato = t2tstr(dato)
    elif isinstance(dato, tuple):
        dato = t2tstr(time.mktime(dato))
    d = dato.split('-')
    for i in [1,2]:
        if len(d[i]) == 1:
            d[i] = '0' + d[i]
    return '-'.join(d)

#--
def remove_duplicates(data):
    if len(data)==0:
        return data
    data = sorted(data, key=lambda x:x[0])
    x = [data[0]]
    for e in data[1:]:
        if e[0] != x[-1][0]:
            x.append(e)
    return x
    
def _get_ty(start, stop):
    elements = ', '.join(['air_temperature',
                          'max(air_temperature PT1H)',
                          'min(air_temperature PT1H)',
                          'sum(precipitation_amount PT1H)'])
    if client_id[0] == None:
        print('Client id for met.no not set. You can get a client ID from ')
        print('https://frost.met.no/auth/requestCredentials.html')
        print('Client id can be then be set with the command')
        print("set_client_id('<your_client_id>')")
        print('For now, if you have a client ID, enter it here:')
        client_id[0] = input("")
    parameters = {
        'sources': 'SN17850',
        #'elements': 'air_temperature',
        'elements': elements,
        'referencetime': '{}/{}'.format(fix_date(start), fix_date(stop))}
    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id[0],''))
    if r.status_code == 200:
        print('Data retrieved from frost.met.no.')
    else:
        print("Didn't retrieve data from frost.met.no.")
        return None
    return get_ty_from_json(r.json())

def get_ty(start, stop, tmax=86400*100):
    # there is a maximum amount of data which can be downloaded in one go,
    # hence tmax. 100 days should be ok.
    start = time.mktime(start)
    stop = time.mktime(stop)
    stop0 = stop
    #tmax = 86400*10
    tyv = []
    stop = 0
    while stop < stop0:
        stop = min(stop0, start+tmax)
        tyv.append(_get_ty(start, stop))
        start = stop
    ty = []
    for new_data in tyv:
        ty = remove_duplicates(ty + new_data)
    return ty
        

    
# todo sjekke overalt at det er riktig aa bruke gmtime -- hva bruker met.no

def make_data_file(start=(2015, 1, 1, 12, 0, 0, 0, 0, 0),
                   stop=(2015, 1, 31, 12, 0, 0, 0, 0, 0)):
    ty = get_ty(start, stop)
    pickle.dump(ty, open(DATA_FILE_NAME, 'wb'))


def get_stored_data():
    try:
        return pickle.load(open(DATA_FILE_NAME, 'rb'))
    except FileNotFoundError:
        print("weather date file %s not found, starting empty"%DATA_FILE_NAME)
        return []

def update_weather_data(stop=None):
    if stop is None:
        stop = time.gmtime()
    old_data = get_stored_data()
    last_date = max([x[0] for x in old_data])
    new_data = get_ty(time.localtime(last_date - 86400), stop)
    updated_data = remove_duplicates(old_data + new_data)
    if updated_data != old_data:
        pickle.dump(updated_data, open(DATA_FILE_NAME, 'wb'))


#make_data_file()

#update_weather_data()
#update_weather_data(stop=(2016, 5, 30, 12, 0,0,0,0,0))

#c = get_stored_data()
#plt.plot(np.diff([x[0] for x in c])/3600)

# def showit(i,m):
#     for n in range(i, i+m):
#         print(time.ctime(c[n][0]))
#         print(c[n][1])

#showit(1000,3)

