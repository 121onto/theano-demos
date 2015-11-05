# Adapted from http://deeplearning.net/tutorial/
from __future__ import (print_function, division)

import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

###########################################################################
## local imports

from utils import initialize_tensor

###########################################################################
## autoencoder

class AutoEncoder(object):

    def __init__(self, np_rng, th_rng=None,
                 input=None, n_visible=784, n_hidden=500,
                 W=None, b_hid=None, b_vis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not th_rng:
            th_rng = RandomStreams(np_rng.randint(2 ** 30))

        if not W:
            W_scale = 4. * np.sqrt(6. / (n_hidden + n_visible))
            W_shape = (n_visible, n_hidden)
            W = theano.shared(
                value = initialize_tensor(W_scale, W_shape, rng=np_rng),
                name='W',
                borrow=True
            )

        if not b_vis:
            b_vis = tehano.shared(
                value = initialize_tensor(0., n_visible, dist='zero'),
                borrow=True
            )

        if not b_hid:
            b_hid = tehano.shared(
                value = initialize_tensor(0., n_hidden, dist='zero'),
                name='b',
                borrow=True
            )

        self.W = W
        self.W_prime = self.W.T
        self.b = b_hid
        self.b_prime = b_vis
        self.th_rng = th_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_valudes(tilde_x)
        z = self.get_reconstructed_input(y)
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1-z), axis=1)
        cost = T.mean(L)

        dparams = T.grad(cost, self.params)
        updates = [(p, p - learning_rate * dp) for p, dp in zip(params, dparams)]

        return (cost, updates)
