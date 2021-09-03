import mxnet as mx
from mxnet import autograd
from mxnet.gluon import HybridBlock
from mxnet.gluon.parameter import Intermediate

def test_retain_grad_drop_grad_gluon2():
    class CompBlock(HybridBlock):
        def __init__(self):
            super().__init__()
            # self.set_intermediate([Intermediate('nleaf_1', None, grad_req='write'),
            #                        Intermediate('nleaf_2', None, grad_req='write')])
            
        def forward(self, a, b):
            out1 = a*b  # out1 is fetched from the dc_ndoutputs NDArray
            self.mark_vars([out1]) # which will temporarily turn off dc.record()
                                         # apply only on hybridize mode
            out2 = out1 * a
            self.mark_vars(out2)
            return out2

    x = mx.np.array([1,2,3,4])
    y = mx.np.array([5,6,7,8])
    x.attach_grad()
    y.attach_grad()
    block2 = CompBlock()
    block2.initialize()
    block2.hybridize() # Future work: unify the frontend call whether hybridize or not
    with autograd.record():
        z = block2(x, y)
    u = block2.get_mark_vars(0)
    z = block2.get_mark_vars([1])
    
    u = block2.get_intermediate('out1')

    u.attach_grad() # Future work: deferredcompute_entry_ should be cleared
    z.attach_grad()
    z.backward(retain_graph=True)

    print('--------------test attach grad-------------')
    print(x.grad, y.grad, u.grad, z.grad)
    assert (u.grad == x).all()
    assert (z.grad == mx.np.array([1,1,1,1])).all()
    assert (x.grad == 2 * x * y).all()
    assert (y.grad == x*x).all()
    print('--------------test drop grad-------------')
    u.drop_grad()
    z.drop_grad()
    y.drop_grad()
    z.backward()
    print(x.grad, y.grad, u.grad, z.grad)
    assert u.grad is None and z.grad is None and y.grad is None
    assert (x.grad == 2 * x * y).all()

if __name__ == '__main__':
    test_retain_grad_drop_grad_gluon2()