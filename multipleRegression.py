'''
Multiple Regressions Classifier Module
y_i = alpha + beta_1*x_i1 + ... ++ beta_n*x_in + error_i
'''
import vectors as vec
import matrix as mat
import statistics as stat
import gradientDescent as gd
import simpleLinearRegression as slg
import dataprocessing as dp
import random

# Predict final label
# beta = [alpha, beta_1, ..., beta_k]
# x_i = [1, x_i1, ..., x_ik]
def predict(x_i, beta):
    return vec.dot(x_i, beta)


# Error function
def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def squaredError(x_i, y_i, beta):
    return error(x_i, y_i, beta) ** 2


# The gradient of the i-th squared error term
def squaredErrorGradient(x_i, y_i, beta):
    return [-2 * x_ij * error(x_i, y_i, beta) for x_ij in x_i]


# Use data too minimize the total error of function over dataset and return coefficients alpha and beta
def estimateCoefficients(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return gd.minimizeStochastic(squaredError,
                                 squaredErrorGradient,
                                 x, y,
                                 0.0001)


# R-Squared
def multipleRSquared(x, y, beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2 for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / slg.totalSumOfSquares(y)