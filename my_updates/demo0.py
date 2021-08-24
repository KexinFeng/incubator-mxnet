# 80 sec for gdb attaching
from mxnet import ndarray as nd
from mxnet import autograd as ag
x = nd.array([1,2,3,4])
x.attach_grad(grad_req="add")
y = nd.array([5,6,7,8])
y.attach_grad()

with ag.record():
    u = x * y
    v = u.detach()
    v.attach_grad()
    z = v * x

z.backward()
print(x.grad)  # x.grad == v
print(v.grad)  # v.grad == x

u.backward(v.grad)
print(x.grad)  # x.grad == y * v.grad = y * x
               # But the true value is x.grad == 2*x*y
print(y.grad)  # y.grad == x * v.grad = x * x