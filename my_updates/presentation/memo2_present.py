from mxnet import ndarray as nd
from mxnet import autograd as ag

@profile
def test_drop_grad():
    x = nd.ones((int(1e3), int(1e4)))*2
    x.attach_grad()
    y = nd.ones((int(1e3), int(1e4)))*5
    y.attach_grad()

    with ag.record():
        u = x*y
        u2 = u.detach()
        z = x*u2

    z.backward()
    assert (x.grad == u2).asnumpy().all()
    u.backward()
    assert (x.grad == y).asnumpy().all()
        
if __name__ == '__main__':
    test_drop_grad()


Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
     4  248.070 MiB  248.070 MiB           1   @profile
     5                                         def test_drop_grad():
     6  291.539 MiB   43.469 MiB           1       x = nd.ones((int(1e3), int(1e4)))*2
     7  329.578 MiB   38.039 MiB           1       x.attach_grad()
     8  367.805 MiB   38.227 MiB           1       y = nd.ones((int(1e3), int(1e4)))*5
     9  405.848 MiB   38.043 MiB           1       y.attach_grad()
    10                                         
    11  405.848 MiB    0.000 MiB           1       with ag.record():
    12  443.992 MiB   38.145 MiB           1           u = x*y
    13  443.992 MiB    0.000 MiB           1           u2 = u.detach()
    14  482.020 MiB   38.027 MiB           1           z = x*u2
    15                                         
    16  482.723 MiB    0.703 MiB           1       z.backward()
    17  483.152 MiB    0.430 MiB           1       assert (x.grad == u2).asnumpy().all()
    18  482.922 MiB   -0.230 MiB           1       u.backward()
    19  483.184 MiB    0.262 MiB           1       assert (x.grad == y).asnumpy().all()