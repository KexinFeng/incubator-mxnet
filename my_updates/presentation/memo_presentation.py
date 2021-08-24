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

    u.attach_grad()
    z.backward(retain_graph=True)
    print(u.grad)
    print('---------test drop grad-------------')
    u.drop_grad()
    y.drop_grad()
    z.backward()
    print(x.grad, y.grad)

if __name__ == '__main__':
    test_drop_grad()


Line #    Mem usage    Increment  Occurences   Line Contents
============================================================
     4  255.500 MiB  255.500 MiB           1   @profile
     5                                         def test_drop_grad():
     6  293.883 MiB   38.383 MiB           1       x = nd.ones((int(1e4), int(1e3)))
     7  332.168 MiB   38.285 MiB           1       x.attach_grad()
     8  370.453 MiB   38.285 MiB           1       y = nd.ones((int(1e4), int(1e3)))
     9  408.223 MiB   37.770 MiB           1       y.attach_grad()
    10                                         
    11  408.223 MiB    0.000 MiB           1       with ag.record():
    12  446.879 MiB   38.656 MiB           1           u = x * y
    13  484.652 MiB   37.773 MiB           1           z = u * x
    14                                         
    15  523.199 MiB   38.547 MiB           1       u.attach_grad()
    16  528.555 MiB    5.355 MiB           1       z.backward(retain_graph=True)
    17  528.551 MiB   -0.004 MiB           1       print(u.grad)
    18  528.551 MiB    0.000 MiB           1       print('---------test drop grad-------------')
    19  490.402 MiB  -38.148 MiB           1       u.drop_grad()
    20  452.254 MiB  -38.148 MiB           1       y.drop_grad()
    21  452.457 MiB    0.203 MiB           1       z.backward()
    22  452.500 MiB    0.043 MiB           1       print(x.grad, y.grad)