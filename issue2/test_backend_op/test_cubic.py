import mxnet as mx
import numpy as np
import numpy.random as rnd
from mxnet.test_utils import *

def rand_shape_nd(num_dim, dim=10):
    return tuple(rnd.randint(1, dim+1, size=num_dim))

def test_cubic_function():

    def f(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    a = np.random.random_sample()
    b = np.random.random_sample()
    c = np.random.random_sample()
    d = np.random.random_sample()
    data = mx.symbol.Variable('data')

    quad_sym = mx.sym.contrib.cubic(data=data, a=a, b=b, c=c, d=d)
    for dtype in [np.float16, np.float32, np.float64]:
        for ndim in range(1, 6):
            shape = rand_shape_nd(ndim, 5)
            data_np = np.random.randn(*shape).astype(dtype)
            expected = f(data_np, a, b, c, d)
            backward_expected = 3*a*data_np**2 + 2*b*data_np + c

            # imperative forward
            output = mx.nd.contrib.cubic(mx.nd.array(data_np), a=a, b=b, c=c, d=d)
            assert_almost_equal(output.asnumpy(), expected, 
                rtol=1e-2 if dtype is np.float16 else 1e-5,
                atol=1e-2 if dtype is np.float16 else 1e-5)

            # forward
            check_symbolic_forward(quad_sym, [data_np], [expected], 
                    rtol=1e-2 if dtype is np.float16 else 1e-5,
                    atol=1e-2 if dtype is np.float16 else 1e-5)

            # backward
            check_symbolic_backward(quad_sym, [data_np], [np.ones(expected.shape)], [backward_expected],
                rtol=1e-2 if dtype is np.float16 else 1e-5,
                atol=1e-2 if dtype is np.float16 else 1e-5)

            # # check backward using finite difference
            # check_numeric_gradient(quad_sym, [data_np], atol=0.001)

if __name__ == '__main__':
    test_cubic_function()
         