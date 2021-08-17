from mxnet import ndarray as nd
from mxnet import autograd as ag
x = nd.array([1,2,3,4])
x.attach_grad()

with ag.record():
    y = x*x
    u = y.detach()
    z = x*u

u.attach_grad()
y.attach_grad()

out_grad = nd.array([10, 10, 10, 10])
z.backward(out_grad)
print(x.grad == u * out_grad)
print(u.grad)
print(z.grad == None)

out_grad = nd.array([0.1, 0.1, 0.1, 0.1])
y.backward(out_grad)
print(x.grad == 2 * x * out_grad)
print(y.grad)

