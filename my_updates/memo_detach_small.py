from mxnet import ndarray as nd
from mxnet import autograd as ag

# @profile
def test_drop_grad():
    x = nd.ones((int(1e0), int(1e1)))*2
    xgrad = nd.ones(x.shape)*7
    ag.mark_variables(x, xgrad)
    # x.attach_grad()
    y = nd.ones((int(1e0), int(1e1)))*5
    y.attach_grad()

    with ag.record():
        u = x*y
        u2 = u.detach()  # making u an output variable
        z = x*u2

    ugrad = nd.ones(x.shape)*7
    ag.mark_variables(u, ugrad)

    z.backward()
    print('------------------------')
    print(xgrad)
    print(x.grad)
    assert (x.grad == u2).asnumpy().all()
    assert z.grad == None

    print('------------------------')
    out_grad = nd.ones(x.shape)*5
    u.backward(out_grad)
    print(ugrad)
    print(u.grad)
    print(out_grad)

    print('------------------------')
    print(xgrad)
    print(x.grad)
    assert (x.grad == out_grad * y).asnumpy().all()

if __name__ == '__main__':
    test_drop_grad()

"""
This test shows bug when applying  ag.mark_variables(u, ugrad)  on output node. The
descrepency between u.grad and ugrad is not expected.
The baviour of     
    xgrad = nd.ones(x.shape)*7
    ag.mark_variables(x, xgrad)
is as expected where xgrad agrees with x.grad

However, conceptually ugrad is not a variable. So it is considered not a proper use
to mark the output u as variable using ag.mark_variable(). But it's ok to attach 
gradient to u.
"""
