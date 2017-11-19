'''
Gradient Descent Module
'''
import vectors as vec
import math
import random

# Step function used by gradient to movie step_size amount in direction from v
def step(v, direction, step_size):
    return [v_i + step_size * dir_i for v_i, dir_i in zip(v, direction)]


# Safety function that gives new function that returns infinity for invalid inputs
def safe(f):
    
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    
    return safe_f


# Batch Gradient Descent implementation to choose parameters which minize the function
def minimizeBatch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    
    # Possible step sizes we can test
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    # Set initial theta
    theta = theta_0

    # safe version of target function
    target_fn = safe(target_fn)

    # the value we are trying to minimize
    value = target_fn(theta)

    # iteratively take steps to minimize the error
    while True:
        gradient = gradient_fn(theta)
        updated_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]

        # choose the one that has minimized error the most
        next_theta = min(updated_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        print "Gradient:", gradient
        print "Old Value:", value
        print "New Value:", next_value

        # stop if converged
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


# Negate helper functions used to negate minimize to maximize
def negate(f):

    # for any input x, return -f(x)
    return lambda *args, **kwargs: -f(*args, **kwargs)


# the sanem but if f returns a list
def negateAll(f):
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]


# Use gradient descent to maximize the function
def maximizeBatch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimizeBatch(negate(target_fn), 
                         negateAll(gradient_fn), 
                         theta_0, 
                         tolerance)


# Helper generator function to return elements of data in random order
def randomizeData(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)

    # randomize indices and return data in new order
    for i in indexes:
        yield data[i]


# Stochastic Gradient Descent implementation
def minimizeStochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    print "Minimize Stochastic"
    data = zip(x, y)

    # initial theta and step size
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float("inf")

    # increment for every iteration with no improvement minimum value
    iterations_with_no_improvement = 0

    # iteratively minimize function at each data point, stop if > 100 iterationswith no improvement
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # if new minimum is found
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # no improvement, shrink step size
            iterations_with_no_improvement += 1
            alpha *= 0.9
        
        # take gradient step for each data point
        for x_i, y_i in randomizeData(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vec.vectorSub(theta, vec.scalarMultiply(alpha, gradient_i))
    
    return min_theta


# Use stochastic gradient descent to maximize function
def maximizeStochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    print "Maximize Stochastic"
    return minimizeStochastic(negate(target_fn),
                              negateAll(gradient_fn),
                              x, 
                              y, 
                              theta_0, 
                              alpha_0)