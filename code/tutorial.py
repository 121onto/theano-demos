from __future__ import (print_function, division)

import numpy as np
import numpy.random as rng
import pandas as pd
import theano
import theano.tensor as T

###########################################################################
## Logistic function
###########################################################################

def logistic_function():
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    logistic = theano.function([x], s)
    print(logistic([[0, 1], [-1, -2]]))
    """
    array([[ 0.5       ,  0.73105858],
           [ 0.26894142,  0.11920292]])
    """

###########################################################################
## Logistic regression
###########################################################################

def logistic_regression(N=500, feats=700, training_steps=10000):
    # random variables
    data = dict(
        x = rng.randn(N, feats),
        y = rng.randint(size=N, low=0, high=2)
    )

    # symbolic variables
    x = T.matrix('x')
    y = T.vector('y')
    w = theano.shared(rng.randn(feats), name='w')
    b = theano.shared(0., name='b')

    # functions
    p_1 = 1/(1 + T.exp(-T.dot(x, w) - b))
    prediction = p_1 > 0.5
    cross_entropy = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
    cost = cross_entropy.mean() + 0.01 * (w ** 2).sum() # regularization
    dw, db = T.grad(cost, [w, b])

    train = theano.function(
        inputs = [x, y],
        outputs = [prediction, cost],
        updates = ((w, w - 0.1 * dw), (b, b - 0.1 * db))
    )
    predict = theano.function(inputs=[x], outputs=prediction)

    pred, err = train(data['x'], data['y'])
    print('Cost before fitting:')
    print(err)

    for i in range(training_steps-1):
        pred, err = train(data['x'], data['y'])

    print('Cost at solution:')
    print(err)
