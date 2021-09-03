from mxnet.ndarray.ndarray import power
import torch as tf
import torch.autograd.functional as fun

from mxnet import autograd as ag
from mxnet import ndarray as nd

print('-------------------------------')
inputs = tf.tensor([[1., 2.], [3., 4.]])
def pow_reducer(x):
    return x.pow(3).sum()
# print(fun.hessian(pow_reducer, inputs))


print('-------------------------------')
x = nd.array([[1, 2], [3, 4]])
x.attach_grad()

with ag.record():
    z = nd.elemwise_add(nd.exp(x), x)
dx, = ag.grad(z, [x], create_graph=True)
dx.backward()
# print(x.grad)

x = nd.array([[1, 2], [3, 4]])
def pow_reducer(x):
    return nd.exp(x)

def hessian(func, inputs):
    inputs.attach_grad()
    with ag.record():
        out = func(inputs)
    dout, = ag.grad(out, [inputs], create_graph=True)
    dout.backward()
    return inputs.grad

res = hessian(pow_reducer, x)
# print(res)


def hessian2(func, inputs):

    inputs.attach_grad()
    for i in range(len(inputs)):
        for j in range(len(inputs[0])):
            inputs[i][j].attach_grad()

    with ag.record():
        out = func(inputs)

    douts = [None for _ in range(len(inputs[0]))] * len(inputs)
    for i in range(len(inputs)):
        for j in range(len(inputs[0])):
            dout_ij, = ag.grad(out, inputs[i][j], create_graph=True)
            douts[i][j] = dout_ij

    res = []
    for i in range(len(inputs)):
        for j in range(len(inputs[0])): 
            douts[i][j].backward()
            res.append(inputs.grad)

    return res

print(hessian2(pow_reducer, x))


