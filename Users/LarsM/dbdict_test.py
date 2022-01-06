
# def make_dbdict(dbfilename, fullfilenames):
#     d = dbdict(dbfilename)
#     for (i, f) in enumerate(fullfilenames):
#         if i%1000 == 0:
#             print(i)
#         try:
#             a = get_data.get_file_data(f)
#         except:
#             print('Failed: ', f)
#             continue
#         aj = json.dumps(a)
#         d[a['filename']] = aj

# def make_dbdict2(dbfilename, fullfilenames):
#     d = dbdict(dbfilename)
#     for (i, f) in enumerate(fullfilenames):
#         if i%1000 == 0:
#             print(i)
#         try:
#             a = get_data.get_file_data(f)
#         except:
#             print('Failed: ', f)
#             continue
#         aj = json.dumps(a)
#         ajc = gzip.compress(aj.encode('utf-8'))
#         d[a['filename']] = ajc

# # %time make_dbdict("testsmalldb", fullrawfiles[:100])
# # %time make_dbdict2("testsmalldb2", fullrawfiles[:100])
# # < 10% forskjell

# %time d2 = dbdict("testdb2")

# def getdata(key):
#     d = dbdict("testdb2")
#     a = d[key]
#     b = gzip.decompress(a)
#     return json.loads(b)

# #%time q=getdata(files[1000]) #2 ms!
# #--

# prover aa lagre raadataene i db, ikke de parsede dataene

def make_or_update_raw_dbdict(dbfilename, fullfilenames):
    d = dbdict(dbfilename)
    keyset = set(d.keys())
    for (i, f) in enumerate(fullfilenames):
        filename = os.path.split(f)[1]
        if i%1000 == 0:
            print(i)
        if filename in keyset:
            continue
        try:
            a = pickle.load(open(f, 'rb'))
        except:
            print('Failed: ', f)
            continue
        aj = json.dumps(a)
        ajc = gzip.compress(aj.encode('utf-8'))
        d[filename] = ajc
    return d

%time rawdb = make_or_update_raw_dbdict("rawdb", fullrawfiles)

keys = rawdb.keys()

#--
def getrawdata(key, d=dbdict("rawdb")):
    a = d[key]
    b = gzip.decompress(a)
    return json.loads(b)

%time a = [get_data.parse_saved_data(getrawdata(key), key) for key in keys[:100]]
%time a = [getrawdata(key) for key in keys[:100]]
# 206 ms and 323 ms. Slower without parsing.

d = dbdict("rawdb")
keys = d.keys()


def getN2O(key):
    a = getrawdata(key)['dlt']
    ty = a['ty']
    t = [x[0] for x in ty]
    y = [x[1][dlt_indexes.N2O_dry] for x in ty]
    return t, y

%time all_n2o = {key:getN2O(key) for key in keys} # ca 45sek
%time all_parsed = [sr.parse_filename(key) for key in keys]
import pylab as plt
import numpy as np
plt.ion()

x = [p['vehicle_pos']['x'] for p in all_parsed]
y = [p['vehicle_pos']['y'] for p in all_parsed]


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

a = [x for x in d.keys() if x.startswith('2021')]
len(a)

for 
