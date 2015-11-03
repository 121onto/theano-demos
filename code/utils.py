from __future__ import (print_function, division)

import cPickle
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
            cPickle.dump(v.get_value(borrow=True), file, -1)

def load_params(params, path):
    with open(path) as file:
        for v in params:
            v.set_value(cPickle.load(file), borrow=True)

###########################################################################
## shared dataset

def shared_dataset(data, borrow=True):
    """
    Function that loads the dataset into shared variables.

    Parameters
    ----------
    data: tuple of numpy arrays
      data = (x, y) where x is an np.array of predictors and y is an np array
      of outcome variables

    """
    x, y = data
    sx = theano.shared(
        np.asarray(x, dtype=theano.config.floatX),
        borrow=borrow
    )
    sy = theano.shared(
        np.asarray(y, dtype=theano.config.floatX),
        borrow=borrow
    )
    return sx, T.cast(sy, 'int32')
