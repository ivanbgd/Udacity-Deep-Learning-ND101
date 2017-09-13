# How to Do Linear Regression using Gradient Descent - Live session from 3/29/17
# https://www.youtube.com/watch?v=XdM6ER7zTLk
# https://github.com/llSourcell/linear_regression_live

# My modification, that uses Numpy to the full extent, which can be faster.

import numpy as np

def computeErrorForGivenPoints(m, b, points):
    x, y = points[:, 0], points[:, 1]
    squareDiff = np.square(y - (m*x + b))
    totalError = squareDiff.mean()
    return totalError

def step_gradient(mCurrent, bCurrent, points, learningRate):
    """ gradient descent """
    x, y = points[:, 0], points[:, 1]
    bGradient = (mCurrent*x + bCurrent) - y
    mGradient = x*bGradient
    mGradient = 2.*mGradient.mean()
    bGradient = 2.*bGradient.mean()
    newM = mCurrent - learningRate*mGradient
    newB = bCurrent - learningRate*bGradient
    return newM, newB

def gradient_descent_runner(points, startingM, startingB, learningRate, numIterations):
    m = startingM
    b = startingB
    for i in range(numIterations):
        m, b = step_gradient(m, b, points, learningRate)
    return m, b

def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    # hyperparameter(s)
    learningRate = .0001
    # y = mx + b (slope formula)
    initialM = 0.
    initialB = 0.
    numIterations = 1000
    print('Starting gradient descent at m = {}, b = {}, error = {}'.format(initialM, initialB, computeErrorForGivenPoints(initialM, initialB, points)))     # error = 5565.1078
    print('Running...')
    m, b = gradient_descent_runner(points, initialM, initialB, learningRate, numIterations)
    print('After {} iterations:'.format(numIterations))
    print('m =', m)     # 1.4777
    print('b =', b)     # 0.0889
    print('error = ', computeErrorForGivenPoints(m, b, points))     # 112.6148


if __name__ == "__main__":
    run()