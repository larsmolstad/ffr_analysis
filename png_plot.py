import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import subprocess
viewer='C:/Users/larsmo/Downloads/JPEGView_1_0_35_1/JPEGView32/JPEGView.exe'
filename = 'c:\\temp\\test.png'

def start():
    subprocess.Popen([viewer, filename])

def show():
    plt.savefig(filename)

start()
