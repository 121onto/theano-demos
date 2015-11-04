# Adapted from http://deeplearning.net/tutorial/
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
## hidden

def initialize_hidden_W_values(scale, n_in, n_out):
    rtn = np.asarray(
        rng.uniform(
            low=-np.sqrt(scale / (n_in + n_out)),
            high=np.sqrt(scale / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    ) * scale
    return rtn

class HiddenLayer(object):
    """
    Notes
    -----
    Transforming the tanh activation function per [LeCun1998] may generate an
    improvement in performance.  The original code did not achieve a validation
    error rate on par with the totorial.  Not sure why.
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            # suggested initial weights larger for sigmoid activiation
            scale = 24. if (activation == theano.tensor.nnet.sigmoid) else 6.
            W = theano.shared(
                value=initialize_hidden_W_values(scale, n_in, n_out),
                name='W',
                borrow=True
            )

        if b is None:
            b = theano.shared(
                value = np.zeros((n_out,), dtype=theano.config.floatX),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else 1.7159 * activation( (2./3) * lin_output) if activation == T.tanh # transformation
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


###########################################################################
## Logistic regression

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

###########################################################################
## mlp

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hidden_layer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        self.log_reg_layer = LogisticRegression(
            input=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.log_reg_layer.W).sum()
        self.L2 = (self.hidden_layer.W ** 2).sum() + (self.log_reg_layer.W ** 2).sum()
        self.negative_log_likelihood = self.log_reg_layer.negative_log_likelihood
        self.errors = self.log_reg_layer.errors

        self.params = self.hidden_layer.params + self.log_reg_layer.params
        self.input = input
        self.y_pred = self.log_reg_layer.y_pred
