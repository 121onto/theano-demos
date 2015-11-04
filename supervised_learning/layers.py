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
from theano.tensor.signal import downsample

###########################################################################
## local imports

from supervised_learning.utils import initialize_tensor

###########################################################################
## hidden

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None:
            # suggested initial weights larger for sigmoid activiation
            W_scale = 24. if (activation == theano.tensor.nnet.sigmoid) else 6.
            W_scale = np.sqrt(W_scale / (n_in + n_out))
            W_shape = (n_in, n_out)
            W = theano.shared(
                value=initialize_tensor(W_scale, W_shape, theano.config.floatX, rng=rng, dist='uniform'),
                name='W',
                borrow=True
            )

        if b is None:
            b_shape = (n_out,)
            b = theano.shared(
                value=initialize_tensor(None, b_shape, theano.config.floatX, rng=rng, dist='zero'),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = b

        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            #else 1.7159 * activation( (2./3) * lin_output) if activation == T.tanh # transformation
            else activation(lin_output)
        )
        self.params = [self.W, self.b]

###########################################################################
## Convolution with optional pooling

class ConvolutionLayer(object):
    """
    A convolution layer with optional pooling using shared valriables
    """
    def __init__(self, rng, input,
                 feature_maps_in, feature_maps_out,
                 filter_shape, input_shape, batch_size,
                 pool_size=None, pool_ignore_border=True,
                 W=None, b=None, activation=T.tanh):

        self.input = input

        W_shape = (feature_maps_out, feature_maps_in) + filter_shape
        b_shape = (feature_maps_out,)
        i_shape = (batch_size, feature_maps_in) + input_shape

        W_scale = 1.0/np.sqrt(feature_maps_in * reduce(np.multiply,filter_shape))
        b_scale = 0.5

        if W is None:
            W = theano.shared(
                initialize_tensor(W_scale, W_shape, rng=rng, dist='uniform'),
                name='W'
            )
        if b is None:
            b = theano.shared(
                initialize_tensor(b_scale, b_shape, rng=rng, dist='zero'),
                name='b'
            )

        self.W = W
        self.b = b

        conv_out = T.nnet.conv.conv2d(
            self.input,
            filters=self.W,
            filter_shape = W_shape,
            image_shape = i_shape
        )

        # pool first to improve performance
        pooled_out = conv_out
        if pool_size is not None:
            pooled_out = downsample.max_pool_2d(
                conv_out,
                pool_size,
                ignore_border=pool_ignore_border
            )

        lin_output = pooled_out + self.b.dimshuffle('x',0,'x','x')
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


###########################################################################
## Max pooling

class MaxPooling(object):
    def __init__(self, input, shape, ignore_border=True):
        self.input = input
        self.output = downsample.max_pool_2d(
            input,
            shape,
            ignore_border=ignore_border
        )

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


###########################################################################
## lenet


class LeNet(object):
    def __init__(self, rng, input,
                 batch_size, n_image_channels, image_size,
                 nkerns, filter_shape, pool_size, n_hidden):

        self.input = input.reshape((batch_size, n_image_channels) + image_size)
        input_shape = image_size
        self.layer0 = ConvolutionLayer(
            rng=rng,
            input=self.input,
            feature_maps_in=n_image_channels,
            feature_maps_out=nkerns[0],
            filter_shape=filter_shape,
            input_shape=input_shape,
            batch_size=batch_size,
            pool_size=pool_size
        )

        input_shape = self.shape_reduction(input_shape, filter_shape, pool_size)
        self.layer1 = ConvolutionLayer(
            rng=rng,
            input=self.layer0.output,
            feature_maps_in=nkerns[0],
            feature_maps_out=nkerns[1],
            filter_shape=filter_shape,
            input_shape=input_shape,
            batch_size=batch_size,
            pool_size=pool_size
        )

        input_shape = self.shape_reduction(input_shape, filter_shape, pool_size)
        self.layer2 = MLP(
            rng=rng,
            input=self.layer1.output.flatten(2),
            n_in=nkerns[1] * reduce(np.multiply, input_shape),
            n_hidden=n_hidden,
            n_out=10
        )

        self.negative_log_likelihood = self.layer2.negative_log_likelihood
        self.errors = self.layer2.errors

        self.params = self.layer0.params + self.layer1.params + self.layer2.params
        self.input = input
        self.y_pred = self.layer2.y_pred


    def shape_reduction(self, input_shape, filter_shape, pool_size):
        rtn = [i-f+1 for i, f in zip(input_shape, filter_shape)]
        rtn = tuple(np.divide(rtn, pool_size))
        return rtn
