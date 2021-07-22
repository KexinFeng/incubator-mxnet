from mxnet import ndarray as nd
from mxnet import autograd as ag
x = nd.array([1,2,3,4])
x.attach_grad()
y = nd.array([5,6,7,8])
y.attach_grad()

ag.set_recording(True)
u = x * y
v = u.detach() 
v.attach_grad() # implicitly run u = u.detach()
z = v * x
ag.set_recording(False)

z.backward()
print(x.grad) # : v = (xy) = [5, 12, 21, 32]
u.backward(v.grad)
print(x.grad, y.grad) # supposed to be [10, 24, 42, 64], [1, 4, 9, 16]
# x.grad : v + v.grad * (y), but in actual output, the first term is missing. 