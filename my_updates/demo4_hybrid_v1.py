import mxnet as mx
from mxnet import autograd
from mxnet.gluon import HybridBlock
from mxnet.gluon.parameter import Intermediate
from numpy.core.shape_base import block

def test_retain_grad_drop_grad_gluon2():
    class CompBlock(HybridBlock):
        def __init__(self):
            super().__init__()

        def forward(self, a, b, c):
            out1 = self.intermediate(('out1_0', 'out1_1'), ((a+b)*c, a*b), grad_req='write')
            out2 = self.intermediate('out2', out1[1] * a)
            return out2

    x = mx.np.array([1,2,3,4])
    y = mx.np.array([5,6,7,8])
    w = mx.np.array([0.1, 0.1, 0.1, 0.1])
    x.attach_grad()
    y.attach_grad()
    w.attach_grad()
    block2 = CompBlock()
    block2.initialize()
    block2.hybridize()
    with autograd.record():
        z = block2(x, y, w)

    block2.attach_grad_intermediate()
    u0 = block2.get_intermediate('out1_0').data()
    u = block2.get_intermediate('out1_1').data()
    z = block2.get_intermediate('out2').data()
    z.backward(retain_graph=True)

    # print('--------------test attach grad-------------')
    print(x.grad, y.grad, u.grad, z.grad, u0.grad)
    assert (u.grad == x).all()
    assert (u0.grad == mx.np.array([0, 0, 0, 0])).all()
    assert (z.grad == mx.np.array([1,1,1,1])).all()
    assert (x.grad == 2 * x * y).all()
    assert (y.grad == x*x).all()
    print('--------------test drop grad-------------')
    u.drop_grad()
    u0.drop_grad()
    z.drop_grad()
    y.drop_grad()
    z.backward()
    print(x.grad, y.grad, u.grad, u0.grad)
    assert u.grad is None and u0.grad is None and y.grad is None and z.grad is None
    assert (x.grad == 2 * x * y).all()

if __name__ == '__main__':
    test_retain_grad_drop_grad_gluon2()