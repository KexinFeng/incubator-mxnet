from mxnet import ndarray as nd
from mxnet import autograd as ag
x = nd.array([1,2,3,4])
x.attach_grad()
y = nd.array([5,6,7,8])
y.attach_grad()


ag.set_recording(True)
u = x * y
u.attach_grad() # implicitly run u = u.detach()
z = u * x
ag.set_recording(False)


print('test1:')
z.backward()
print('x.grad', x.grad, '\ny.grad', y.grad) 
# supposed to be: (v = xy = [5, 12, 21, 32], [0, 0, 0, 0] or x**2 ?)
print('u.grad', u.grad) # supposed to be: x = [1, 2, 3, 4]
print('')


print('test2')
u.backward(u.grad, retain_graph=True)
print('x.grad', x.grad, '\ny.grad', y.grad) 
# supposed to be: (x + v.grad * y = [10, 24, 42, 64], [0, 0, 0, 0] or x**2 ?)
print('u.grad', u.grad)
print('')


print('test3')
u.backward()
print('x.grad', x.grad, '\ny.grad', y.grad)
print('u.grad', u.grad) # supposed to be x = [1, 2, 3, 4]
