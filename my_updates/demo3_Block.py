import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, HybridBlock

def test_retain_grad_drop_grad_gluon():
    class MulBlock(HybridBlock):
        def __init__(self):
            super().__init__()
        def forward(self, a, b):
            return a * b
    
    input1 = mx.np.array([1,2,3,4])
    input2 = mx.np.array([5,6,7,8])
    input1.attach_grad()
    input2.attach_grad()
    block1 = MulBlock()
    block1.initialize()
    block1.hybridize()
    with autograd.record():
        output1 = block1(input1, input2)
        output2 = block1(output1, input1)  # output2 = x * (x*y)
    output1.attach_grad()
    output2.backward()
    print(input1.grad, input2.grad, output1.grad)

def test_retain_grad_drop_grad_gluon2():
    # class MulBlock(HybridBlock):
    #     def __init__(self):
    #         super().__init__()
    #     def forward(self, a, b):
    #         return a * b
    class CompBlock(HybridBlock):
        def __init__(self):
            super().__init__()
            self.marked_var = None
        def forward(self, a, b):
            out1 = a*b
            out2 = out1 * a
            self.marked_var = out1
            return out2
    x = mx.np.array([1,2,3,4])
    y = mx.np.array([5,6,7,8])
    x.attach_grad()
    y.attach_grad()
    block2 = CompBlock()
    block2.initialize()
    # block2.hybridize()
    with autograd.record():
        z = block2(x, y)
    u = block2.marked_var
    u.attach_grad()
    z.attach_grad()
    z.backward(retain_graph=True)
    print(x.grad, y.grad, block2.marked_var.grad)

    assert (u.grad == x).all()
    assert (z.grad == mx.np.array([1,1,1,1])).all()
    assert (x.grad == 2 * x * y).all()
    assert (y.grad == x*x).all()

    u.drop_grad()
    z.drop_grad()
    y.drop_grad()
    z.backward()
    print(x.grad, y.grad, u.grad)

    assert u.grad is None and z.grad is None and y.grad is None
    assert (x.grad == 2 * x * y).all()

if __name__ == '__main__':
    test_retain_grad_drop_grad_gluon()
    test_retain_grad_drop_grad_gluon2()