from mxnet import ndarray as nd
from mxnet import autograd as ag

# @profile
def test_drop_grad():
    x = nd.ones((int(1e3), int(1e4)))*2
    x.attach_grad()
    y = nd.ones((int(1e3), int(1e4)))*5
    y.attach_grad()

    with ag.record():
        u = x*y
        u2 = u.detach()
        ztmp = x*u2
        z = ztmp*x

    # ugrad = nd.ones(x.shape)*3
    # ugrad = nd.zeros(x.shape)
    # ag.mark_variables(u, ugrad)
    u.attach_grad()

    # del ztmp, u2
    # del ugrad
    z.backward()
    
    u.backward()
    del u
    del x


if __name__ == '__main__':
    test_drop_grad()

"""
Similar to the problem in `memo_detach_small`,  u.backward() causes memory increase
when u is marked:    
    ugrad = nd.ones(x.shape)*3
    ugrad = nd.zeros(x.shape)
    ag.mark_variables(u, ugrad)
But u.backward() does not cause memory increase if u.attach_grad() is called.
This behaviour is not expected.
"""