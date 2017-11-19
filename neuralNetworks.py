'''
Artificial Neural Networks Module
'''
import vectors as vec
import matrix as mat
import statistics as stat
import gradientDescent as gd
import dataprocessing as dp
import random
import math

# Step function
def step(x):
    return 1 if x >= 0 else 0

# Sigmoid function
def sigmoid(t):
    return 1 / (1 + math.exp(-t))


# Calculate the neuron's output
# weights = [w_i, ..., w_n, bias]
# inX = [x_1, ..., x_n, 1]
def neuronOutput(weights, inX):
    return sigmoid(vec.dot(weights, inX))


# Feed Forward Neural Network
# Represented as a list (layers) of lists (neurons) of lists (weights)
def feedForward(neuralNetwork, inX):

    # takes given neural network and returns output
    outputs = []

    # process one layer at a time
    for layer in neuralNetwork:
        inputVec = inX + [1]
        output = [neuronOutput(neuron, inputVec) for neuron in layer]

        outputs.append(output)

        # output is the input vector of next layer
        inputVec = output
    
    return outputs