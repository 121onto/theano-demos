from __future__ import (print_function, division)

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

###########################################################################
## Logistic function
###########################################################################

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)
logistic([[0, 1], [-1, -2]])
"""
array([[ 0.5       ,  0.73105858],
       [ 0.26894142,  0.11920292]])
"""
