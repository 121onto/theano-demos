# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import timeit
import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T

###########################################################################
## local imports

from utils import load_data, display_results
from layers import AutoEncoder
from solvers import UnsupervisedMSGD

###########################################################################
## config

SEED = 1234

###########################################################################
## main

def fit_autoencoder(image_size=(28, 28), n_image_channels=1,
            datasets='../data/mnist.pkl.gz', outpath='../output/mnist_autoencoder.params',
            n_visible=784, n_hidden=500,
            learning_rate=0.01, corruption_level=0.0,
            n_epochs=1000, batch_size=20, patience=10000,
            patience_increase=2, improvement_threshold=0.995):

    index = T.lscalar()
    x = T.dmatrix(name='input')

    encoder = AutoEncoder(
        np_rng=rng.RandomState(SEED),
        input=x,
        th_rng=None,
        n_visible=n_visible,
        n_hidden=n_hidden,
        corruption_level=corruption_level
    )
    learner = UnsupervisedMSGD(
        index,
        x,
        batch_size,
        learning_rate,
        load_data(datasets),
        outpath,
        encoder,
        encoder.cost
    )
    best_validation_error, best_iter, epoch, elapsed_time = learner.fit(
        n_epochs=n_epochs,
        patience=patience,
        patience_increase=patience_increase,
        improvement_threshold=improvement_threshold
    )
    display_results(best_validation_error, elapsed_time, epoch)

    return learner


if __name__ == '__main__':
    fit_autoencoder(corruption_level=0.0)
    fit_autoencoder(corruption_level=0.3)
