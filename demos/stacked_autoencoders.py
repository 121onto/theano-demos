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
from layers import StackedAutoEncoder
from solvers import SupervisedMSGD, UnsupervisedMSGD

###########################################################################
## config

SEED = 1234

###########################################################################
## main

def fit_stacked_autoencoder(image_size=(28, 28), n_out=10,
            datasets='../data/mnist.pkl.gz', outpath='../output/mnist_autoencoder.params',
            hidden_layer_sizes=[500, 500, 500], corruption_levels=[0.1, 0.2, 0.3],
            learning_rate_encoder=0.001, learning_rate_full=0.1,
            n_epochs_encoder=15, n_epochs_full=1000,
            batch_size_encoder=1, batch_size_full=20,
            patience_encoder=5000, patience_full=5000,
            patience_increase=2, improvement_threshold=0.995):

    n_inputs = reduce(np.multiply, image_size)
    index = T.lscalar(name='input')
    x = T.matrix(name='x')
    y = T.ivector(name='y')
    datasets = load_data(datasets)

    stacked_encoder = StackedAutoEncoder(
        x=x,
        y=y,
        np_rng=rng,
        th_rng=None,
        n_inputs=n_inputs,
        hidden_layer_sizes=hidden_layer_sizes,
        corruption_levels=corruption_levels,
        n_out=n_out
    )

    # pretrain
    for i, encoder in enumerate(stacked_encoder.encoder_layers):
        print("Pre-training encoder layer %i" % i)
        learner = UnsupervisedMSGD(
            index,
            x,
            batch_size_encoder,
            learning_rate_encoder,
            datasets,
            None,
            encoder,
            encoder.cost
        )
        best_validation_error, best_iter, epoch, elapsed_time = learner.fit(
            n_epochs=n_epochs_encoder,
            patience=patience_encoder,
            patience_increase=patience_increase,
            improvement_threshold=improvement_threshold
        )
        print("resuts for pre-training encoder %i" % i)
        display_results(best_validation_error, elapsed_time, epoch)

    print("Fitting full model")
    learner = SupervisedMSGD(
        index,
        x,
        y,
        batch_size_full,
        learning_rate_full,
        datasets,
        outpath,
        stacked_encoder,
        stacked_encoder.cost
    )
    best_validation_error, best_iter, epoch, elapsed_time = learner.fit(
        n_epochs=n_epochs_full,
        patience=patience_full,
        patience_increase=patience_increase,
        improvement_threshold=improvement_threshold
    )
    display_results(best_validation_error, elapsed_time, epoch)


if __name__ == '__main__':
    fit_stacked_autoencoder(corruption_levels=[0.1,0.2, 0.3])
