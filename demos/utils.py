# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T

###########################################################################
## i/o

def load_data(path='data/mnist.pkl.gz'):
    train, valid, test = None, None, None
    with gzip.open(path, 'rb') as f:
        train, valid, test = cPickle.load(f)

    tn_x, tn_y = make_shared(train)
    v_x , v_y  = make_shared(valid)
    tt_x, tt_y = make_shared(test)
    return [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)]


def make_shared(data, borrow=True):
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


def display_results(best_validation_loss, elapsed_time, epoch):
    print(
        'Optimization complete with best validation score of %f %%'
        % (best_validation_loss * 100.)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec'
        % (epoch, 1. * epoch / (elapsed_time))
    )
