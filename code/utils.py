from __future__ import (print_function, division)

import numpy as np
import numpy.random as rng
import pandas as pd
import theano
import theano.tensor as T

###########################################################################
## i/o

def save_params(params, path):
    with open(path, 'wb') as file:
        for v in params:
            cPickle.dump(v.get_value(borrow=True), save_file, -1)

def load_params(params, path):
    with open(path) as file:
        for v in params:
            v.set_value(cPickle.load(save_file), borrow=True)

###########################################################################
## shared dataset

def shared_dataset(data):
    """
    Function that loads the dataset into shared variables.

    Parameters
    ----------
    data: tuple of numpy arrays
      data = (x, y) where x is an np.array of predictors and y is an np array
      of outcome variables

    """
    x, y = data
    sx = theano.shared(np.asarray(x, dtype=theano.config.floatX))
    sy = theano.shared(np.asarray(y, dtype=theano.config.floatX))
    return sx, T.cast(sy, 'int32')


def test_shared_dataset():
    train, valid, test = None, None, None
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train, valid, test = cPickle.load(f)

    tn_x, tn_y = shared_dataset(train)
    v_x , v_y  = shared_dataset(valid)
    tt_x, tt_y = shared_dataset(test)
