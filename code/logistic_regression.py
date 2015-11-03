from __future__ import (print_function, division)

import timeit
import gzip
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
## Logistic function

def logistic_function():
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    logistic = theano.function([x], s)
    print(logistic([[0, 1], [-1, -2]]))
    """
    array([[ 0.5       ,  0.73105858],
           [ 0.26894142,  0.11920292]])
    """

###########################################################################
## Logistic regression with random data

def logistic_regression_random_data(N=500, feats=700, training_steps=10000):
    # data
    data = dict(
        x = rng.randn(N, feats),
        y = rng.randint(size=N, low=0, high=2)
    )

    # symbolic variables
    x = T.matrix('x')
    y = T.vector('y')
    w = theano.shared(rng.randn(feats), name='w')
    b = theano.shared(0., name='b')

    # functions
    p_1 = 1/(1 + T.exp(-T.dot(x, w) - b))
    prediction = p_1 > 0.5
    cross_entropy = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
    cost = cross_entropy.mean() + 0.01 * (w ** 2).sum() # regularization
    dw, db = T.grad(cost, [w, b])

    train = theano.function(
        inputs = [x, y],
        outputs = [prediction, cost],
        updates = ((w, w - 0.1 * dw), (b, b - 0.1 * db))
    )
    predict = theano.function(inputs=[x], outputs=prediction)

    pred, err = train(data['x'], data['y'])
    print('Cost before fitting:')
    print(err)

    for i in range(training_steps-1):
        pred, err = train(data['x'], data['y'])

    print('Cost at solution:')
    print(err)

###########################################################################
## Logistic regression from tutorial

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        """
        Parameters
        ----------
        n_in: tuple
          the dimention of the input
        n_out: integer
          the number of classes
        """
        self.W = theano.shared(
            value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        Parameters
        ----------
        y: theano.tensor.TensorType
          corresponds to a vecotr that gievs for each example the correct label

        Notes
        -----
        Use mean rather than sum so the learning rate is less dependent on the
        batch size
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch; zero one
        loss over the size of the minibatch.

        Parameters
        ----------
        y: theano.tensor.TensorType
          corresponds to a vecotr that gievs for each example the correct label
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset_path='data/mnist.pkl.gz'):
    train, valid, test = None, None, None
    with gzip.open(dataset_path, 'rb') as f:
        train, valid, test = cPickle.load(f)

    tn_x, tn_y = shared_dataset(train)
    v_x , v_y  = shared_dataset(valid)
    tt_x, tt_y = shared_dataset(test)
    return [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)]


def fit_msgd_early_stopping(datasets, n_batches, models, classifier,
                   n_epochs=1000, patience=5000, patience_increase=2,
                   improvement_threshold=0.995):

    # unpack parameters
    [(tn_x, tn_y), (v_x, v_y), (tt_x, tt_y)] = datasets
    n_tn_batches, n_v_batches, n_tt_batches = n_batches
    tn_model, v_model, tt_model = models

    validation_frequency = min(n_tn_batches, patience/2)

    # initialize some variables
    best_validation_loss = np.inf
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


            if patience <= iter:
                done_looping = True
                break

        save_params(classifier.params, path='output/mnist_logistic_regression.params')

    end_time = timeit.default_timer()
    return best_validation_loss, epoch, (end_time - start_time)

###########################################################################
## fit and predict

def fit_logistic(learning_rate=0.13, n_epochs=1000, dataset='data/mnist.pkl.gz', batch_size=600):
    # load data
    datasets = load_data(dataset)
    tn_x, tn_y = datasets[0]
    v_x, v_y = datasets[1]
    tt_x, tt_y = datasets[2]

    n_tn_batches = int(tn_x.get_value(borrow=True).shape[0] / batch_size)
    n_v_batches = int(v_x.get_value(borrow=True).shape[0] / batch_size)
    n_tt_batches = int(tt_x.get_value(borrow=True).shape[0] / batch_size)
    n_batches = [n_tn_batches, n_v_batches, n_tt_batches]

    # define symbolic parameters, graphs, and updates
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    cost = classifier.negative_log_likelihood(y)
    dW = T.grad(cost=cost, wrt=classifier.W)
    db = T.grad(cost=cost, wrt=classifier.b)
    updates = [
        (classifier.W, classifier.W - learning_rate * dW),
        (classifier.b, classifier.b - learning_rate * db)
    ]

    # compile theano functions
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
    models = [tn_model, v_model, tt_model]

    best_validation_loss, epoch, elapsed_time = fit_msgd_early_stopping(
        datasets,
        n_batches,
        models,
        classifier
    )

    print(
        'Optimization complete with best validation score of %f %%'
        % (best_validation_loss * 100.)
    )
    print(
        'The code run for %d epochs, with %f epochs/sec'
        % (epoch, 1. * epoch / (elapsed_time))
    )


def predict():
    # load classifier
    x = T.matrix('x')
    y = T.ivector('y')
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    load_params(classifier.params, path='output/mnist_logistic_regression.params')

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
