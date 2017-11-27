treatments_small_plot_numbers = \
        {101:'Control', 105:'Dolomite', 109:'Norite', 115:'Marble',
         121:'Olivine', 123:'Larvikite', 129:'Olivine', 131:'Norite',
         303:'Larvikite', 307:'Dolomite', 311:'Control', 313:'Olivine',
         317:'Marble', 319:'Norite', 325:'Larvikite', 327:'Dolomite',
         501:'Marble', 503:'Control', 509:'Norite', 513:'Dolomite',
         519:'Control', 523:'Larvikite', 525:'Marble', 531:'Olivine'}

    
bucket_numbers = {
    1:131,
    2:129,
    3:123,
    4:121,
    5:115,
    6:109,
    7:105,
    8:101,
    9:303,
    10:307,
    11:311,
    12:313,
    13:317,
    14:319,
    15:325,
    16:327,
    17:531,
    18:525,
    19:523,
    20:519,
    21:513,
    22:509,
    23:503,
    24:501}


def bucket_treatment(nr):
    return treatments_small_plot_numbers[bucket_numbers[nr]]

def bucket_treatment_numbers():
    a = {}
    for t in sorted(list(set([bucket_treatment(x) for x in range(1,25)]))):
        a[t] = [x for x in range(1,25) if bucket_treatment(x)==t]
    return a

def show_bucket_treatment_numbers():
    a = bucket_treatment_numbers()
    for key, val in a.items():
        print(key, val)
        
bucket_treatments = {key: bucket_treatment(key) for key in bucket_numbers}
