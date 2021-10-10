import json
import gzip
import os
import pickle
from dbdict import dbdict
import sys
sys.path.append('/home/larsmo/div/ffr/merge/ffr_analysis/prog')
import dlt_indexes

def write(a, filename):
    json_object = json.dumps(a)
    gzip.open(filename, "w").write(json_object.encode('utf-8'))


def read(filename):
    s = gzip.open(filename).read()
    return json.loads(s)
    
# %time write(a, "a.gz") #tar under 4 ms
# %time b=read("a.zip") #tar under 2 ms

rawdir = '/home/larsmo/div/ffr/merge/_RAWDATA'
rawdir2 = os.path.join(rawdir, '..', '_RAWDATA_JSON_GZ')
files = os.listdir(rawdir)
len(files)
files = [x for x in files if x.startswith('2')]
len(files)
fullrawfiles = [os.path.join(rawdir, x) for x in files]
fullgzfiles = [os.path.join(rawdir2, x+'.gz') for x in files]

#files = [os.path.join(rawdir, x) for x in files]


def translate_file(filename):
    full_filename = os.path.join(rawdir,filename)
    a = get_data.get_file_data(full_filename)
    write(a, os.path.join(rawdir2, filename+".gz"))

#translate_file(files[0])

q = read(os.path.join(rawdir2, files[0]+".gz"))

# failed = []
# for (i,f) in enumerate(files):
#     if i%1000 == 0:
#         print(i)
#     try:
#         translate_file(f)
#     except:
#         failed.append(f)


def testraw(n1, n2):
    for i in range(n1, n2):
        a = get_data.get_file_data(fullrawfiles[i])

def testjz(n1, n2):
    for i in range(n1, n2):
        a = read(fullgzfiles[i])

#%time testraw(10000,10100)
#%time testjz(10000,10100)

def make_dbdict(dbfilename, fullfilenames):
    d = dbdict(dbfilename)
    for (i, f) in enumerate(fullfilenames):
        if i%1000 == 0:
            print(i)
        try:
            a = get_data.get_file_data(f)
        except:
            print('Failed: ', f)
            continue
        aj = json.dumps(a)
        d[a['filename']] = aj

def make_dbdict2(dbfilename, fullfilenames):
    d = dbdict(dbfilename)
    for (i, f) in enumerate(fullfilenames):
        if i%1000 == 0:
            print(i)
        try:
            a = get_data.get_file_data(f)
        except:
            print('Failed: ', f)
            continue
        aj = json.dumps(a)
        ajc = gzip.compress(aj.encode('utf-8'))
        d[a['filename']] = ajc

# make_dbdict("testdb", fullrawfiles)
# %time make_dbdict("testsmalldb", fullrawfiles[1:100])
# %time make_dbdict2("testdb2", fullrawfiles)


%time d2 = dbdict("testdb2")

def getdata(key):
    d = dbdict("testdb2")
    a = d[key]
    b = gzip.decompress(a)
    return json.loads(b)

#%time q=getdata(files[1000]) #2 ms!
#--
# prover aa lagre raadataene i db, ikke de parsede dataene
def update_dbdict3(dbfilename, fullfilenames):
    d = dbdict(dbfilename)
    keys = d.keys()
    for (i, f) in enumerate(fullfilenames):
        filename = os.path.split(f)[1]
        if filename in keys:
            continue
        if i%1000 == 0:
            print(i)
        try:
            a = pickle.load(open(f, 'rb'))
        except:
            print('Failed: ', f)
            continue
        aj = json.dumps(a)
        ajc = gzip.compress(aj.encode('utf-8'))
        d[filename] = ajc
    return d

%time rawdb = update_dbdict3("rawdb", fullrawfiles)
keys = rawdb.keys()
#--

def update_dbdict3_faster(dbfilename, fullfilenames):
    d = dbdict(dbfilename)
    fullkeys = [os.path.join(os.path.split(fullfilenames[1])[0], key) for key in keys]
    newfullrawfiles = set(fullfilenames) - set(fullkeys)
    print("number of new files:", len(newfullrawfiles))
    update_dbdict3(dbfilename, newfullrawfiles)
    return d

#--
def getrawdata(key):
    d = dbdict("rawdb")
    a = d[key]
    b = gzip.decompress(a)
    return json.loads(b)

#%time a = [getrawdata(key) for key in keys[:100]]

d = dbdict("rawdb")
keys = d.keys()


def getN2O(key):
    a = getrawdata(key)['dlt']
    ty = a['ty']
    t = [x[0] for x in ty]
    y = [x[1][dlt_indexes.N2O_dry] for x in ty]
    return t, y

# getN2O(keys[1000])

all_n2o = {key:getN2O(key) for key in keys}
#--
import pylab as plt
import numpy as np
plt.ion()
plt.plot([1])
hikeys = set([])
#plt.clf()
for key in keys:
    t,y = all_n2o[key]
    if np.mean(y) > 5 and max(y)<30 and key > '2020-11-01'  and max(t)<500: 
        print(key)
        hikeys.add(key)
        plt.plot(t, y)#, color='black')
        plt.pause(0.01)

#--

for key in keys:
    t, y = all_n2o[key]
    start = '2020-10-01'
    stop = '2020-12-01'
    if key > start and key < stop:
        print(key)

#--
