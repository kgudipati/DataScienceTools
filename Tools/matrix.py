'''
Matrix Arithmatic Module
'''

# Get row i of the given nxk matrix A
def getRow(A, i):
    return A[i]


# Get column i of given matrix
def getColumn(A, i):
    return [A_i[i] for A_i in A] 


# Make a matrix given shape and function to generate elements
def createMatrix(rows, cols, fn):
    return [[fn(i, j) for j in range(cols)] for i in range(rows)]