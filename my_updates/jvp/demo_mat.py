import torch
import numpy as np

from mxnet import ndarray as nd
from mxnet import autograd as ag

print('---------------------------------------')
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
y = torch.tensor([5., 6., 7., 8.], requires_grad=True)
u = x * y
u.requires_grad_(True)
u.retain_grad()
z = x * u
z.sum().backward()
print(x.grad, y.grad, u.grad)


print('-----------------------------------------')

x = nd.array([[1,2],[3,4]])
x.attach_grad()
y = nd.array([[5,6],[7,8]])
y.attach_grad()

with ag.record():
    u = x * y
    z = u * x

u.attach_grad()
# z.attach_grad()

out_grad = nd.array([[10, 10], [10, 10]])
z.backward(out_grad, retain_graph=True)

print(x.grad, y.grad, u.grad)
# print('u', u.grad)
# print('z', z.grad)
# print(x.grad, y.grad)

# assert (u.grad == out_grad * x).asnumpy().all()
# assert (z.grad == out_grad).asnumpy().all()
# assert (y.grad == out_grad * x*x).asnumpy().all()

# print('---------test drop grad-------------')
# u.drop_grad()
# z.drop_grad()
# y.drop_grad()

# out_grad = nd.array([[0.1, 0.1], [0.1, 0.1]])
# z.backward(out_grad)

# print('u', u.grad)
# print('z', z.grad)
# print('x', x.grad)
# print('y', y.grad)

# assert u.grad is None and z.grad is None
# assert (x.grad == out_grad * 2 * x * y).asnumpy().all()
# assert y.grad is None