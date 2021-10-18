import numpy as np
from bisect_find import bisect_find
from scipy import stats

# storing the regression results in a namedtuple
# se_intercept and se_slope are the standard errors
# mse mean standard error?
# start and stop: if not using the whole set


class Regression():
    
    def __init__(self,
                 intercept,
                 slope,
                 se_intercept,
                 se_slope,
                 mse,
                 start,
                 stop,
                 rsq,
                 pval,
                 min_y,
                 max_y):
        self.intercept = intercept
        self.slope = slope
        self.se_intercept = se_intercept
        self.se_slope = se_slope
        self.mse = mse
        self.start = start
        self.stop = stop
        self.rsq = rsq
        self.pval = pval
        self.min_y = min_y
        self.max_y = max_y

    def set_start_and_stop(self, start, stop):
        self.start = start
        self.stop = stop

    def __str__(self):
        def f(x):
            return '{:.5g}'.format(x)
        s = 'Regr(slope:{}, intercept: {}, se_intercept: {}, se_slope: {}, mse: {}'
        s += ', start: {}, stop: {}, rsq: {}, pval: {}, min_y: {}, max_y: {})'
        return s.format(f(self.slope), f(self.intercept), f(self.se_intercept),
                        f(self.se_slope), f(self.mse), self.start, self.stop,
                        self.rsq, self.pval, self.min_y, self.max_y)

    def __repr__(self):
        return self.__str__()

    def to_dict(self): # for saving as json later todo
        return {x: get(self, x) for x in
                ('intercept', 'slope', 'se_intercept', 'se_slope', 'mse',
                 'start', 'stop', 'rsq', 'pval', 'min_y', 'max_y')}

    
def mean(x):
    return sum(x) * 1.0 / len(x)

#Performs the actual regression!
def regression2(x, y, plotfun=False):
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]  #Least Squares 
    #    slope, intercept = np.polyfit(x, y, 1)
    ymod = intercept + x * slope
    n = len(x) # EEB: number of points in the final regression? Is it already trimmed to only those points before it gets to regression2?
    mse = sum((ymod - y)**2) / (n - 2)
    mx = mean(x)
    xc = x - mx
    xcxc = np.dot(xc, xc)
    se_intercept = mse * np.sqrt(1.0 / n + mx**2 / xcxc)
    se_slope = mse * np.sqrt(1.0 / xcxc)
    min_y = min(y)
    max_y = max(y)
        
    # EEB More stats:  t-test, P value, R squared
    # never got stats.t.cdf working, use pearsonr function instead
    #var_x = np.var(x,ddof=1)
    #var_y = np.var(y,ddof=1)
    #stddev=np.sqrt((var_x + var_y)/2)
    #tstat = (x.mean() - y.mean())/(stddev*np.sqrt(2/n))
    #df = 2*n - 2
    #pval = 1 - stats.t.cdf(tstat,df)
    r, pval = stats.pearsonr(y,x) # get R and p-value
    rsq=r*r #r-squared
    if plotfun:
        plotfun(x, y, '.', x, intercept + slope * x)
    return Regression(intercept, slope, se_intercept, se_slope, mse, x[0], x[-1], rsq, pval, min_y, max_y)


def find_best_regression(x, y, xint, crit='mse', jump=1, plotfun=False):
    """Finds the best (mse-wise) or steepest regression line (if
     crit=='steepest') with largest x-difference between first and last
     point (because of the switching). Multiplying the distance with
     the slope or dividing the mse by the distance -- actually not
     distance: distance - xint*.3. Todo explain why.

    """
    def bisect_find_next(i):
        """ adds jump to i, finds next j such that
        x[j+1] >= x[i]+xint and j >= i+2 """
        j = 0
        while j > -1 and j < i + 2 and i + jump < len(x) - 1:
            i += jump
            j = bisect_find(x, x[i] + xint)
        return i, (j if j >= i + 2 else -1)
    if len(x) == 0 or xint > x[-1] - x[0]:
        return None
    bestmse_x_span = 1e99
    bestb1_x_span = -1e99
    x = np.array(x)
    y = np.array(y)
    i, j = bisect_find_next(i=-1)
    best, besti, bestj = None, 0, 0
    while j > -1:
        xspan = max(1e-99, (x[j] - x[i] - xint * 0.3))
        reg = regression2(x[i:j], y[i:j])
        if ((reg.mse * xspan < bestmse_x_span and crit == 'mse')
                or (reg.slope * xspan > bestb1_x_span and crit == 'steepest')):
            besti, bestj = i, j
            best = reg
            bestmse_x_span = reg.mse * xspan
            bestb1_x_span = reg.slope * xspan
        i, j = bisect_find_next(i)
    if plotfun and best:
        bestx = x[besti:bestj]
        plotfun(x, y, '.',
                bestx, best.intercept + best.slope * bestx, '-')
    best.set_start_and_stop(x[besti], x[bestj])
    #print('finished find_best_regression EEB')
    return best

#Selects data points that will be included in the regression. 
#This runs on one side, one gas at a time ... the data has already been split into sides by now.  
def regress_within(x, y, x1, x2, plotfun=False):
    #Safety checks to make sure the range of regression is within range of the data. x1 and x2 are start/stop points
    i = 0 if x1 <= x[0] else bisect_find(x, x1, True)
    j = len(x) - 1 if x2 >= x[-1] else bisect_find(x, x2, True)
    if i < 0:
        raise Exception('x1=%g before x[0]=%g' % (x1, x[0]))
    if j < 0:
        raise Exception('x2=%g after x[-1]=%g' % (x2, x[-1]))
    # make arrays with possible valid x (time) and y(measurement) values
    xin = np.array(x)[i:j]
    yin = np.array(y)[i:j]
    #Calls regression2 with selected points
    reg = regression2(xin, yin)
    if plotfun:
        plotfun(x, y, '.', xin, reg.intercept + xin * reg.slope)
    reg.set_start_and_stop(x[i], x[j])
    #print('finished regress_within EEB')
    return reg
