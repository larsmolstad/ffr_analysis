import math
import numpy as np
from collections import namedtuple
from bisect_find import bisect_find

# storing the regression results in a namedtuple
# se_intercept and se_slope are the standard errors
# mse mean standard error?
# start and stop: if not using the whole set


class Regression():
    def __init__(self, intercept, slope, se_intercept, se_slope, mse, start, stop):
        self.intercept = intercept
        self.slope = slope
        self.se_intercept = se_intercept
        self.se_slope = se_slope
        self.mse = mse
        self.start = start
        self.stop = stop

    def set_start_and_stop(self, start, stop):
        self.start = start
        self.stop = stop

    def __str__(self):
        def f(x):
            return '{:.5g}'.format(x)
        s = 'Regr(slope:{}, intercept: {}, se_intercept: {}, se_slope: {}, mse: {}, start: {}, stop: {})'
        return s.format(f(self.slope), f(self.intercept), f(self.se_intercept),
                        f(self.se_slope), f(self.mse), self.start, self.stop)

    def __repr__(self):
        return self.__str__()


def mean(x):
    return sum(x) * 1.0 / len(x)


def scapr(x, y):
    """scalar product of x and y"""
    return sum([x[i] * y[i] for i in range(len(x))])


def centralize(x):
    return [a - mean(x) for a in x]


# some different versions of linear regression. Using regression2

def regression(x, y):
    """ returns b0, b1, seb1, seb2, mse,
    where seb1 and seb2 are standard errors
    b0 is intercept, b1 is slope"""
    n = len(x)
    xc = centralize(x)
    yc = centralize(y)
    mx, my = mean(x), mean(y)
    xcxc = scapr(xc, xc)
    xcyc = scapr(xc, yc)
    b1 = xcyc / xcxc
    b0 = my - b1 * mx
    mse = sum([(b0 + b1 * x[i] - y[i])**2 for i in range(n)]) / (n - 2)
    seb0 = mse * math.sqrt(1.0 / n + (mx**2) / xcxc)
    seb1 = mse * math.sqrt(1.0 / xcxc)
    return Regression(b0, b1, seb0, seb1, mse, 0, n - 1)


def regression2(x, y, plotfun=False):
    slope, intercept = np.polyfit(x, y, 1)
    ymod = intercept + x * slope
    n = len(x)
    mse = sum((ymod - y)**2) / (n - 2)
    mx = mean(x)
    my = mean(y)
    xc = x - mx
    xcxc = np.dot(xc, xc)
    se_intercept = mse * np.sqrt(1.0 / n + mx**2 / xcxc)
    se_slope = mse * np.sqrt(1.0 / xcxc)
    if plotfun:
        plotfun(x, y, '.', x, intercept + slope * x)
    return Regression(intercept, slope, se_intercept, se_slope, mse, 0, n - 1)


def regression3(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return m, c


def find_best_regression(x, y, xint, cr='best', jump=1, plotfun=False):
    """Finds the best (mse-wise) or steepest regression line (if
     cr=='steepest') with largest x-difference between first and last
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
        if ((reg.mse * xspan < bestmse_x_span and cr == 'best')
                or (reg.slope * xspan > bestb1_x_span and cr == 'steepest')):
            besti, bestj = i, j
            best = reg
            bestmse_x_span = reg.mse * xspan
            bestb1_x_span = reg.slope * xspan
        i, j = bisect_find_next(i)
    if plotfun and best:
        bestx = x[besti:bestj]
        plotfun(x, y, '.',
                bestx, best.intercept + best.slope * bestx, '-')
    best.set_start_and_stop(besti, bestj)
    return best


def regress_within(x, y, x1, x2, plotfun=False):
    i = 0 if x1 <= x[0] else bisect_find(x, x1, True)
    j = len(x) - 1 if x2 >= x[-1] else bisect_find(x, x2, True)
    if i < 0:
        raise Exception, 'x1=%g before x[0]=%g' % (x1, x[0])
    if j < 0:
        raise Exception, 'x2=%g after x[-1]=%g' % (x2, x[-1])
    xin = np.array(x)[i:j]
    yin = np.array(y)[i:j]
    reg = regression2(xin, yin)
    if plotfun:
        plotfun(x, y, '.', xin, reg.intercept + xin * reg.slope)
    reg.set_start_and_stop(i, j)
    return reg

