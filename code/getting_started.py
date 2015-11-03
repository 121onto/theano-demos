# Adapted from http://deeplearning.net/tutorial/gettingstarted.html#gettingstarted
from __future__ import (print_function, division)

import cPickle
import numpy as np
import numpy.random as rng
import pandas as pd
import theano
import theano.tensor as T

###########################################################################
## local imports

from code.utils import (shared_dataset, load_params, save_params)

###########################################################################
## minibatch SGD pseudo code

def minibatch_sgd(loss, params, learning_rate):
    d_loss_wrt_params = T.grad(loss, params)
    updates = [(params, params - learning_rate * d_loss_wrt_params)]
    MSGD = theano.function([x_batch,y_batch], loss, updates=updates)

    for (x_batch, y_batch) in train_batches:
        MSGD(x_batch, y_batch)
        if stopping_condition_is_met:
            return params

###########################################################################
## early stopping pseudo code

def early_stopping(loss, data, patience=5000, patience_increase=2, improvement_threshold=0.995):
    validation_frequency = min(n_train_batches, patience/2)

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            # d_loss_wrt_params = ... # compute gradient
            # params -= learning_rate * d_loss_wrt_params # gradient descent

            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:

                # this_validation_loss = ... # compute loss on validation set
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_params = copy.deepcopy(params)
                    best_validation_loss = this_validation_loss

            if patience <= iter:
                done_looping = True
                break

        save_params(params)
