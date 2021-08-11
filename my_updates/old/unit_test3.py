print('z = x* (x*y)')
import torch
import numpy as np
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
y = torch.tensor([5., 6., 7., 8.], requires_grad=True)
u = x * y
u.requires_grad_(True)
u.retain_grad()
z = x * u
z.sum().backward()
# u.retain_grad()
print(x.grad, y.grad, u.grad)


print('\nmxnet')
from mxnet import autograd as ag
from mxnet import ndarray as nd
x = nd.array([1,2,3,4])
x.attach_grad()
y = nd.array([5,6,7,8])
y.attach_grad()

with ag.record():
    u = x * y
    u.attach_grad()
    z = x * u
    
z.backward()
print(x.grad, y.grad, u.grad) 
# supposed to be [10., 24., 42., 64.], [1.,  4.,  9., 16.], x = [1,2,3,4]


print('-------------------')
print('z = 2(5* x**2 + 13x + 10), x = x1 + x2**2')
import torch
import numpy as np
x1 = torch.tensor([1.,2.,3.,4.], requires_grad=True)
x2 = torch.tensor([5.,6.,7.,8.], requires_grad=True)
x = x1 + x2**2
y = 5 * (x**2) + (13 * x) + 10
y.requires_grad_(True)
y.retain_grad()
z = 2 * y
z.sum().backward()
print(x1.grad, x2.grad, y.grad)


import mxnet as mx
x1 = mx.nd.array([1.,2.,3.,4.], ctx = mx.cpu())
x2 = mx.nd.array([5.,6.,7.,8.], ctx = mx.cpu())
x1.attach_grad()
x2.attach_grad()
with mx.autograd.record():
    x = x1 + x2**2
    y = ((5 * (x**2)) + (13 * x) + 10)
    y.attach_grad()
    z = 2 * y
z.backward()
print(x1.grad, x2.grad, y.grad)
y.backward(y.grad)
print(x1.grad, x2.grad, y.grad)


