
import json
import gzip
import os
example_file = '../../example_data/2017-11-15-14-38-02-x599255_904103-y6615141_29233-z0_0-h-2_39656403177_right_Plot_22_'
a = get_data.get_file_data(example_file)

def write(a, filename):
    json_object = json.dumps(a)
    gzip.open(filename, "w").write(json_object.encode('utf-8'))


def read(filename):
    s = gzip.open(filename).read()
    return json.loads(s)
    
# %time write(a, "a.gz") #tar under 4 ms
# %time b=read("a.zip") #tar under 2 ms

rawdir = '/home/larsmo/div/ffr/compare_larsmolstad_and_erin/merge_test/_RAWDATA'
rawdir2 = os.path.join(rawdir, '..', '_RAWDATA_JSON_GZ')
files = os.listdir(rawdir)
len(files)
files = [x for x in files if x.startswith('2')]
len(files)
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

fullrawfiles = [os.path.join(rawdir, x) for x in files]
fullgzfiles = [os.path.join(rawdir2, x+'.gz') for x in files]

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

#make_dbdict("testdb", fullrawfiles)
%time make_dbdict("testsmalldb", fullrawfiles[1:100])
%time make_dbdict2("testdb2", fullrawfiles)


%time d2 = dbdict("testdb2")

def getdata(key, dictname="testdb2"):
    d = dbdict("testdb2")
    a = d[key]
    b = gzip.decompress(a)
    return json.loads(b)

#%time q=getdata(files[1000]) #2 ms!
