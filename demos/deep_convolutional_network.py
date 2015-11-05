# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import numpy as np
import numpy.random as rng
import theano.tensor as T

###########################################################################
## local imports

from utils import load_data, display_results
from layers import LeNet
from solvers import MiniBatchSGD

###########################################################################
## config

SEED = 1234

###########################################################################
## main

def fit_lenet(image_size=(28, 28), n_image_channels=1,
              datasets='../data/mnist.pkl.gz', outpath='../output/mnist_lenet.params',
              filter_shape=(5, 5), nkerns=(2, 6), pool_size=(2,2), n_hidden=500,
              learning_rate=0.01, L1_reg=0.00, L2_reg=0.001,
              n_epochs=1000, batch_size=20, patience=10000,
              patience_increase=2, improvement_threshold=0.995):


    # build model
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LeNet(
        rng=rng.RandomState(SEED),
        input=x,
        batch_size=batch_size,
        n_image_channels=n_image_channels,
        image_size=image_size,
        nkerns=nkerns,
        filter_shape=filter_shape,
        pool_size=pool_size,
        n_hidden=n_hidden
    )
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2
    )
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
    lenet = fit_lenet()
    print(lenet.predict('test'))
