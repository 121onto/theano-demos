from __future__ import (print_function, division)

from theano import function, config, shared, sandbox, tensor, Out
import numpy
import time

def run_test():
    vlen = 10 * 30 * 700
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f1 = function(
        [],
        tensor.exp(x)
    )
    f2 = function(
        [],
        Out(
            tensor.exp(x),
            borrow=True
        )
    )


    t0 = time.time()
    for i in xrange(iters):
        r = f1()
    t1 = time.time()
    no_borrow = t1 - t0
    t0 = time.time()
    for i in xrange(iters):
        r = f2()
    t1 = time.time()

    print('Looping', iters, 'times took', no_borrow, 'seconds without borrow', end='')
    print('and', t1 - t0, 'seconds with borrow.')


if __name__ == "__main__":
    run_test()
