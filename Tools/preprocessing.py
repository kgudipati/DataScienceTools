'''
Data Processing Module
'''
import matrix as mat
import statistics as stat
import math

# Given multiple dimensions in input data (nxk matrix) 
# Get correlation between each feature
def correlationMatrix(data):
    
    _, num_cols = shape(data)

    def correlationMatrixElement(i, j):
        # get the correlation betwen i and j dimensions
        return stat.correlation(mat.getColumn(data, i), mat.getColumn(data, j))

    return mat.createMatrix(num_cols, num_cols, correlationMatrixElement)


# Rescale data so mean of 0 and st dev of 1
def scale(data):
    num_rows, num_cols = shape(matrix)
    
    # get means and standard deviations of every vector in given data matrix
    means = [stat.mean(mat.getColumn(matrix, j)) for j in range(num_cols)]
    stdevs = [stat.standardDeviation(getColumn(matrix, j)) for j in range(num_cols)]

    return means, stdevs

def rescale(data):
    
    # get means and stdevs of data
    means, stdevs = scale(data)

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data[i][j] - means[j) / stdevs[j]
        else:
            return data[i][j]
    
    num_rows, num_cols = shape(data)

    # create and return new scaled matrix
    return mat.createMatrix(num_rows, num_cols, rescaled)