import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet import autograd as ag

class AddBlock(HybridBlock):
    def __init__(self):
        super(AddBlock, self).__init__()
    
    def forward(self, a, b):
        c = a + b
        return c

add = AddBlock()
add.initialize()
add.hybridize(static_alloc=False)

x = mx.np.array([0.4])
y = mx.np.array([0.5])
x.attach_grad(grad_req='null')
y.attach_grad(grad_req='write')

ag.set_recording(True)
# with ag.record():
out = add(x, y)
    # out = add(x, y)
ag.set_recording(False)
out.backward()

print(x, y, out)
print(x.grad)
print(y.grad)
print(out.grad)
# print("\nINPUT 1: {}\nINPUT 2: {}\nOUTPUT: {}\nGRAD 1: {}\n"
#     "GRAD 2: {}\n".format(x, y, out, x.grad, y.grad, out.grad))
