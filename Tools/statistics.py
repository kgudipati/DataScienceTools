'''
Statistics Module
'''
import vectors as vec
import collections
import math

# Get mean of given list of numbers
def mean(x):
    return sum(x)/len(x)


# Get median
def median(v):
    n = len(v)
    sorted_v = sorted(v)
    mid = n//2
    
    if n % 2 == 1:
        # return middle value if odd length
        return sorted_v[mid]
    else:
        # return the average of the middle values
        lo = mid - 1
        hi = mid
        return (sorted_v[lo] + sorted_v[hi]) / 2


# Get quantile, value less than which a certain percentile of data lies within
# the p-th percentile value in x
def quantile(x, p):
    p_idx = int(p * len(x))
    return sorted(x)[p_idx]


# Get the most common values
def mode(x):
    counts = collections.Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.iteritems() if count == max_count]


# The data range, simple dispersion calculation
def data_range(x):
    return max(x) - min(x)


# Variance: how a single variable deviaes from its means
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    n = len(x)
    deviations = de_mean(x)

    return vec.sumOfSquares(deviations)/(n - 1)


# Standard Deviation
def standardDeviation(x):
    return math.sqrt(variance(x))


# A more robust dispersion calculation, unaffected by a small # of outliers
def interquantileRange(x):
    return quantile(x, 0.75) - quantile(x, 0.25)


# Covariance: how two variables vary in tandem from their means
# Large Positive: x is large when y is large, x is small when y is small
# Large Negative: x and y are opposite
# 0: no relationship between data
def covariance(x, y):
    n = len(x)
    return vec.dot(de_mean(x), de_mean(y)) / (n - 1)


# Correlation between data sets
# -1: anti-correlation 1: perfect correlation
def correlation(x, y):
    stdev_x = standardDeviation(x)
    stdev_y = standardDeviation(y)

    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # no variation