'''
Vector Arithmatic Module
'''
import math

# Component-wise vector addition
def vectorAdd(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]


# Component-wise vector subtraction
def vectorSub(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]


# Sum of all given vectors
def vectorSum(vectors):
    return reduce(vectorAdd, vectors)


# Multiply a vector by a scalar
def scalarMultiply(c, v):
    return [c * v_i for v_i in v]


# Compoonent-wise means of he given vectors
def vectorMean(vectors):
    n = len(vectors)
    return scalarMultiply(1/n, vectorSum(vectors))


# Dot Product
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


# Vector sum of squares
def sumOfSquares(v):
    return dot(v, v)


# Magnitude of vector
def magnitude(v):
    return math.sqrt(sumOfSquares(v))


# Distance between two vectors
def distance(v, w):
    return magnitude(vectorSub(v, w))
