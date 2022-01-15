# %% Imports:
import sys
import os
import time
import glob
from collections import namedtuple
import numpy as np
import pylab as plt
import pandas as pd
import textwrap
pd.options.mode.chained_assignment = None
sys.path.append(os.path.realpath(os.path.join(os.getcwd(), '../../prog')))
import resdir
import get_data
import utils
import regression
import find_regressions
import sort_results as sr
import divide_left_and_right
import weather_data
import flux_calculations
import ginput_show
import polygon_utils
# from polygon_utils import plot_rectangles
# import scipy.stats
from statsmodels.formula.api import ols#, rlm
# from statsmodels.stats.anova import anova_lm
# import statsmodels.api as sm
# from scipy.stats import norm
import xlwt
import shutil
import errno
import re
import datetime
 
fixpath = utils.ensure_absolute_path
