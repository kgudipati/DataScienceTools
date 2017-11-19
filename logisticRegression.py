'''
Logistic Regression Classifier Module
'''
import vectors as vec
import matrix as mat
import statistics as stat
import gradientDescent as gd
import simpleLinearRegression as slg
import dataprocessing as dp
import random
import math

# Logistic Function 
def logistic(x):
    return 1.0 / (1 + math.exp(-x))

def logisticDerivative(x):
    return logistic(x) * (1 - logistic(x))


def logisticLogLikelihood_(x_i, y_i, beta):
    if y_i == 1:
        return math.log(logistic(vec.dot(x_i, beta)))
    else:
        return math.log(1 - logistic(vec.dot(x_i, beta)))

def logisticLogLikelihood(x, y, beta):
    return sum(logisticLogLikelihood_(x_i, y_i, beta) for x_i, y_i in zip(x, y))

def logisticLogPartial_(x_i, y_i, beta, j):
    return (y_i - logistic(vec.dot(x_i, beta))) * x_i[j]

def logisticLogGradient_(x_i, y_i, beta):
    return [logisticLogPartial_(x_i, y_i, beta, j) for j, _ in enumerate(beta)]

def logisticLogGradient(x, y, beta): 
    return reduce(vec.vectorAdd, [logisticLogGradient_(x_i, y_i, beta) for x_i, y_i in zip(x,y)])
