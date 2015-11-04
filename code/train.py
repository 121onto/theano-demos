from __future__ import (print_function, division)

import gzip
import cPickle
import numpy as np
import numpy.random as rng
import pandas as pd
import theano
import theano.tensor as T

###########################################################################
## local imports

from code.utils import (load_params, save_params,
                        shared_dataset,
                        fit_msgd_early_stopping)

from code.layers import (HiddenLayer, LogisticRegression, MLP)

###########################################################################
## config

SEED = 1234

###########################################################################
## i/o

def load_data(dataset_path='data/mnist.pkl.gz'):
    train, valid, test = None, None, None
    with gzip.open(dataset_path, 'rb') as f:
        train, valid, test = cPickle.load(f)

    tn_x, tn_y = shared_dataset(train)
    v_x , v_y  = shared_dataset(valid)
    tt_x, tt_y = shared_dataset(test)
    return [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)]


def display_results(best_validation_loss, elapsed_time, epoch):
    print(
        'Optimization complete with best validation score of %f %%'
        % (best_validation_loss * 100.)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec'
        % (epoch, 1. * epoch / (elapsed_time))
    )


###########################################################################
## helpers

def compute_n_batches(tn, v, tt, batch_size=20):
    n_tn_batches = int(tn.get_value(borrow=True).shape[0] / batch_size)
    n_v_batches = int(v.get_value(borrow=True).shape[0] / batch_size)
    n_tt_batches = int(tt.get_value(borrow=True).shape[0] / batch_size)
    return [n_tn_batches, n_v_batches, n_tt_batches]

def compile_models(datasets, inputs, outputs, updates, index, batch_size):
    tn_x, tn_y = datasets[0]
    v_x, v_y = datasets[1]
    tt_x, tt_y = datasets[2]

    tn_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: tn_x[index * batch_size: (index + 1) * batch_size],
            y: tn_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    v_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        updates=updates,
        givens={
            x: v_x[index * batch_size: (index + 1) * batch_size],
            y: v_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    tt_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: tt_x[index * batch_size: (index + 1) * batch_size],
            y: tt_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    return [tn_model, v_model, tt_model]

###########################################################################
## fit

def fit_logistic(learning_rate=0.13, n_epochs=1000, dataset='data/mnist.pkl.gz', batch_size=600):
    # load data
    datasets = load_data(dataset)
    tn_x, tn_y = datasets[0]
    v_x, v_y = datasets[1]
    tt_x, tt_y = datasets[2]

    n_batches = compute_n_batches(tn_x, v_x, tt_x, batch_size)

    # define symbolic variables
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # define symbolic theano functions
    classifier = LogisticRegression(
        input=x,
        n_in=28*28,
        n_out=10
    )
    cost = classifier.negative_log_likelihood(y)
    dW = T.grad(cost=cost, wrt=classifier.W)
    db = T.grad(cost=cost, wrt=classifier.b)
    updates = [
        (classifier.W, classifier.W - learning_rate * dW),
        (classifier.b, classifier.b - learning_rate * db)
    ]

    # compile theano functions
    models = compile_models(dataset, inputs, outputs, index, batch_size)

    # compute solution
    best_validation_loss, best_iter, epoch, elapsed_time = fit_msgd_early_stopping(
        datasets,
        n_batches,
        models,
        classifier,
        'output/mnist_logistic_regression.params'
    )

    display_results(best_validation_loss, elapsed_time, epoch)


def fit_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.001,
            n_epochs=1000, dataset='data/mnist.pkl.gz',
            batch_size=20, n_hidden=500):

    # load data
    datasets = load_data(dataset)
    tn_x, tn_y = datasets[0]
    v_x, v_y = datasets[1]
    tt_x, tt_y = datasets[2]
    n_batches = compute_n_batches(tn_x, v_x, tt_x, batch_size)

    # define symbolic variables
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # define symbolic theano functions
    classifier = MLP(
        rng=rng.RandomState(SEED),
        input=x,
        n_in=28*28,
        n_hidden=n_hidden,
        n_out=10
    )
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2
    )
    dparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (p, p - learning_rate * dp)
        for p, dp in zip(classifier.params, dparams)
    ]

    # compile theano functions
    models = compile_models(dataset, inputs, outputs, index, batch_size)

    # compute solution
    best_validation_loss, best_iter, epoch, elapsed_time = fit_msgd_early_stopping(
        datasets,
        n_batches,
        models,
        classifier,
        'output/mnist_mlp.params',
        patience=10000
    )

    display_results(best_validation_loss, elapsed_time, epoch)


###########################################################################
## predict

def predict(dataset='data/mnist.pkl.gz', param_path='output/mnist_logistic_regression.params'):
    # load classifier
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    load_params(classifier.params, path=param_path)

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    datasets = load_data(dataset)
    tt_x, tt_y = datasets[2]
    tt_x = tt_x.get_value()

    predicted_values = predict_model(tt_x[:10])
    print ("Predicted values for the first 10 examples in test set:")
    print(predicted_values)
