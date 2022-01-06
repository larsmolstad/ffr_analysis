import json
import gzip
import os
import pickle
from dbdict import dbdict
import sys
sys.path.append('/home/larsmo/div/ffr/merge/ffr_analysis/prog')
import dlt_indexes

def write_jsongz(a, filename):
    json_object = json.dumps(a)
    gzip.open(filename, "w").write(json_object.encode('utf-8'))


def read_gzfile(filename):
    s = gzip.open(filename).read()
    return json.loads(s)
    
# %time write_jsongz(a, "a.gz") #tar under 4 ms
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

def jsonzipfilename(filename, with_dir=True):
    filename = filename+".gz"
    return os.path.join(rawdir2, filename) if with_dir else filename

def translate_file(filename):
    full_filename = os.path.join(rawdir,filename)
    a = get_data.get_file_raw_data(full_filename)
    write_jsongz(a, jzfilename(filename))

# translate_file(files[0])

q = read(os.path.join(rawdir2, files[0]+".gz"))

try:
    failed
except:
    failed = []
 
def translate_files(files):
    nok, nfail, nwas = 0,0,0
    done = os.listdir(rawdir2)
    for (i,f) in enumerate(files):
        if i%1000 == 0:
            print(i)
        jzname = jsonzipfilename(f, False)
        if jzname in done:
            nwas += 1
            continue
        try:
            translate_file(f)
            nok += 1
        except:
            failed.append(f)
            nfail += 1
    print("{} translated, {} already done, {} failed".format(nok, nwas, nfail))
    
translate_files(files)

def testraw(n1, n2):
    for i in range(n1, n2):
        a = get_data.get_file_data(fullrawfiles[i])

def testjz(n1, n2):
    for i in range(n1, n2):
        a = read(fullgzfiles[i])

#%time testraw(10000,10100)
#%time testjz(10000,10100)

#--
"""
b = get_data.get_file_data(fullgzfiles[1000])
name = os.path.splitext(os.path.split(fullgzfiles[1000])[1])[0]
a = get_data.get_file_data(name)
a['filename'] == os.path.splitext(b['filename'])[0]
for key in a.keys():
    print("{}\t{}".format(key, a[key] == b[key]))
a['side']
b['side']
"""

#--
