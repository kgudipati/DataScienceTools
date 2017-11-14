'''
Statistics Module
'''

# Get mean of given list of numbers
def mean(x):
    return sum(x)/len(x)


# Get median
def median(v):
    n = len(v)
    sorted_v = sorted(v)
    mid = n//2


# Make a matrix given shape and function to generate elements
def createMatrix(rows, cols, fn):
    return [[fn(i, j) for j in range(cols)] for i in range(rows)]