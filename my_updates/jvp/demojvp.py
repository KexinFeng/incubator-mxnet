import torch as tf
import torch.autograd.functional as fun

from mxnet import autograd as ag
from mxnet import ndarray as nd

print('-------------------------------')
def exp_reducer(x):
    return x.exp().sum(dim=1)

inputs = tf.tensor([[1., 2.], [3., 4.]])
print(inputs.sum(dim=1))

jac = fun.jacobian(exp_reducer, inputs)
print(jac)


print('-------------------------------')
x = nd.array([[1, 2], [3, 4]])
def exp_reducer(x):
    return nd.sum(nd.exp(x), axis=1)

def jacobian(func, inputs):
    out = func(inputs)
    res = []
    for i in range(len(out)):
        inputs.attach_grad()
        with ag.record():
            out_i = func(inputs)[i]
        out_i.backward()
        res.append(inputs.grad)
    return res

res = jacobian(exp_reducer, x)
print(res[0],'\n', res[1])




