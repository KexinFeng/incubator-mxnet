from mxnet import ndarray as nd
from mxnet import autograd as ag

@profile
def test_drop_grad():
    x = nd.ones((int(1e4), int(1e3)))
    x.attach_grad()
    y = nd.ones((int(1e4), int(1e3)))
    y.attach_grad()

    with ag.record():
        u = x * y
        z = u * x

    # u.attach_grad():
    ugrad = nd.zeros(x.shape, stype=None, dtype=x.dtype)
    ag.mark_variables(u, ugrad)

    z.attach_grad()
    z.backward(retain_graph=True)

    print('---------test drop grad-------------')
    u.drop_grad()
    del ugrad
    z.drop_grad()
    y.drop_grad()
    del y
    z.backward()

    del z
    del x

if __name__ == '__main__':
    test_drop_grad()