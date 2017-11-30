"""This is to be able to switch between my-plotter, which I use from
Emacs, and pylab, which cannot be used from Emacs, but which works
in for example Spyder.

"""

try:
    import justatest #removing justatest.py to test with pylab...
    import my_plotter2 as mp
    plt = mp
except:
    import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D

