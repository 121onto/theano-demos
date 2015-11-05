# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import numpy as np
import numpy.random as rng
import theano.tensor as T

###########################################################################
## local imports

from utils import load_data, display_results
from layers import LogisticRegression
from solvers import MiniBatchSGD

###########################################################################
## config

SEED = 1234

###########################################################################
## main

def fit_logistic(image_size=(28, 28),
             datasets='../data/mnist.pkl.gz', outpath='../output/mnist_logistic_regression.params',
             learning_rate=0.13, n_epochs=1000, batch_size=600,
             patience=5000, patience_increase=2, improvement_threshold=0.995):

    # build model
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')


    # define symbolic theano functions
    classifier = LogisticRegression(
        input=x,
        n_in=reduce(np.multiply, image_size),
        n_out=10
    )
    cost = classifier.negative_log_likelihood(y)
    learner = MiniBatchSGD(
        index,
        x,
        y,
        batch_size,
        learning_rate,
        load_data(datasets),
        outpath,
        classifier,
        cost
    )

    best_validation_loss, best_iter, epoch, elapsed_time = learner.fit(
        n_epochs=n_epochs,
        patience=patience,
        patience_increase=patience_increase,
        improvement_threshold=improvement_threshold
    )

    display_results(best_validation_loss, elapsed_time, epoch)

    return learner


if __name__ == '__main__':
    logistic = fit_logistic()
    print(logistic.predict('test'))
