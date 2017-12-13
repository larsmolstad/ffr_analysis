""" binary search with bisect """

import bisect

def bisect_find(x, xi, nearest=False):
    """finds biggest i such that x[i]<=xi
If nearest is True, return i+1 if xi is closer to x[i+1]"""
    
    if xi<x[0] or xi>x[-1]:
        return -1        
    try:
        i = bisect.bisect(x, xi) - 1
    except:
        return -1
    if i + 1 < len(x) and (x[i+1]==xi or (nearest and abs(x[i+1] - xi) < abs(x[i] - xi))):
        i += 1
    return i
