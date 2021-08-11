from mxnet import ndarray as nd
from mxnet import autograd as ag
import mxnet as mx

def test_gradient():
    x = mx.nd.ones((1,))
    x.attach_grad()
    print('reach here')

    with mx.autograd.record():
        z = mx.nd.elemwise_add(mx.nd.exp(x), x)
    dx, = mx.autograd.grad(z, [x], create_graph=True)
    assert abs(dx.asscalar() - 3.71828175) < 1e-7
    dx.backward()
    assert abs(x.grad.asscalar() - 2.71828175) < 1e-7

if __name__ == '__main__':
    test_gradient()