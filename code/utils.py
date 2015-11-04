from __future__ import (print_function, division)

import timeit
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

###########################################################################
## solvers

def fit_msgd_early_stopping(datasets, n_batches, models, classifier, outpath,
                   n_epochs=1000, patience=5000, patience_increase=2,
                   improvement_threshold=0.995):

    # unpack parameters
    [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)] = datasets
    n_tn_batches, n_v_batches, n_tt_batches = n_batches
    tn_model, v_model, tt_model = models

    validation_frequency = min(n_tn_batches, patience/2)

    # initialize some variables
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    # main loop
    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_tn_batches):
            minibatch_avg_cost = tn_model(minibatch_index)
            iter = (epoch - 1) * n_tn_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [v_model(i) for i in xrange(n_v_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_tn_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    #best_params = copy.deepcopy(params)
                    best_validation_loss = this_validation_loss
                    best_iter = iter


            if patience <= iter:
                done_looping = True
                break

        save_params(classifier.params, path=outpath)

    end_time = timeit.default_timer()
    return best_validation_loss, best_iter, epoch, (end_time - start_time)
