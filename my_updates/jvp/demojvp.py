import torch as tf
import torch.autograd.functional as fun
import numpy as np

arr = np.array([[1, 2], [3, 4]])
# print(np.sum(arr, axis=0))
# print(np.sum(arr, axis=1))

def exp_reducer(x):
    return x.exp().sum(dim=1)
inputs = tf.tensor([[1., 2.], [3., 4.]])
print(inputs.sum(dim=1))

jac = fun.jacobian(exp_reducer, inputs)
print(jac)


#----------------------------------
from mxnet import ndarray as nd
from mxnet import autograd as ag
x = nd.array([1,2,3,4])
x.attach_grad()



