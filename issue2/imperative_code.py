from mxnet import ndarray as nd
from mxnet import numpy as np
from mxnet import autograd as ag
x = nd.array([0.4])
# x.attach_grad()
y = nd.array([0.5])
y.attach_grad()

ag.set_recording(True)
z = x + y
ag.set_recording(False)

print('forward output: ')
print('z= ', z)
print('backward ouput:')
z.backward(nd.array([70]))
print('grad_arrays: \n', x.grad, y.grad)
  