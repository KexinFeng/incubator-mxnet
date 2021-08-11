# 80 sec for gdb attaching
from mxnet import ndarray as nd
from mxnet import autograd as ag
x = nd.array([1,2,3,4])
x.attach_grad()
y = nd.array([5,6,7,8])
y.attach_grad()

with ag.record():
    u = x * y
    # u.retain_grad()
    # v = u.detach() 
    # u.attach_grad() # implicitly run u = u.detach()
    z = u * x
    # u.retain_grad()

u.retain_grad()
z.retain_grad()
out_grad = nd.array([10, 10, 10, 10])
z.backward(out_grad)

assert (u.grad == out_grad * x).asnumpy().all()             # u.grad = out_grad * x
assert (z.grad == out_grad).asnumpy().all()                 # z.grad = out_grad
assert (x.grad == out_grad * 2 * x * y).asnumpy().all()     # x.grad = 2*x*y; y.grad = x**2
assert (y.grad == out_grad * x*x).asnumpy().all() 

# u.backward(u.grad)
# print(x.grad, y.grad) # supposed to be [10, 24, 42, 64], [1, 4, 9, 16]
# x.grad : v + v.grad * (y), but in actual output, the first term is missing. 