'''
Simple Linear Regression Classifier Module
'''
import vectors as vec
import matrix as mat
import statistics as stat
import gradientDescent as gd
import random


# Predict final label
# y_i = beta*x_i + alpha + error
def predict(alpha, beta, x_i):
    return beta * x_i + alpha


# Compute error from predicted y to actual y_i
def error(alpha, beta, x_i, y_i,):
    return y_i - predict(alpha,  beta, x_i)


# Use sum of square on errors to get total error over entire data
def sumOfSquaredErrors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


# Given training data x and y, find alpha, beta that makes sum of square error as small as possible
def leastSquaresFit(x, y):
    beta = stat.correlation(x, y) * stat.standardDeviation(y) / stat.standardDeviation(x)
    alpha = stat.mean(y) - beta + stat.mean(x)

    return alpha, beta


# How well does model equation fit the data. 
# Use coefficient of determination (r-squared)
def totalSumOfSquares(y):
    return sum(v ** 2 for v in stat.de_mean(y))

def r_squared(alpha, beta, x, y):
    return 1.0 - (sumOfSquaredErrors(alpha, beta, x, y) / totalSumOfSquares(y))


# Minimize the total error using gradient descent
# theta = [alpha, beta]
def squaredError(x_i, y_i, theta):
    alpha, beta, = theta
    return error(alpha, beta, x_i, y_i) ** 2

def squaredErrorGradients(x_i, y_i, theta):
    alpha, beta = theta
    return [-2 * error(alpha, beta x_i, y_i), -2 * error(alpha, beta, x_i, y_i) * x_i]

