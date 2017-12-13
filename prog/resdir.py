""" providing names of directories where the data files and results are stored"""

import os
path = os.path.split(os.getcwd())
if path[1] == 'sort_ffr_results':
    raw_data_path = os.path.join(path[0], 'results')
    slopes_path = path[0]
else:
    raw_data_path = os.path.join(os.getcwd(), 'results')
    slopes_path = os.getcwd()
