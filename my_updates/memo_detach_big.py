from mxnet import ndarray as nd
from mxnet import autograd as ag

@profile
def test_drop_grad():
    x = nd.ones((int(1e3), int(1e4)))*2
    x.attach_grad()
    y = nd.ones((int(1e3), int(1e4)))*5
    y.attach_grad()

    with ag.record():
        u = x*y
        u2 = u.detach()
        z = x * u2

    ugrad = nd.ones(x.shape)*3
    ag.mark_variables(u, ugrad)
    # u.attach_grad()

    z.backward()  # z_ograds memo is allocated and released
    u.backward()  # u_ograds memo is allocated and attached to
                  # u.info.out_grads while the original 
                  # u.info.out_grads is not released since it's 
                  # referenced by ugrad
    del x  # release both the Chunk in x (info.outputs) and 
           # info.out_grads 
    print(u.grad)
    del u  # release the memo in info.out_grads
    del u2  # release the Chunk that was shared by u and u2

if __name__ == '__main__':
    test_drop_grad()


"""
# Test resultss
Similar to the problem in `memo_detach_small`,  u.backward() causes memory increase
when u is marked:    
    ugrad = nd.ones(x.shape)*3
    ugrad = nd.zeros(x.shape)
    ag.mark_variables(u, ugrad)
But u.backward() does not cause memory increase if u.attach_grad() is called.
This behaviour is not expected.
"""

"""
# Memory behaviour
Call u.backward() and ograd is not provided. Then in Imperative::Backward(), new array is 
allocated as ograd, which causes memo increase. But if u.attach_grad() is called, then
the attached memory in u.autograd_entry_.node->info.out_grads will be released since it 
will be updated to be ograd. So there is no memo increase in the line u.backward(). But if
u is marked by ag.mark_variables(u, ugrad), then memo in u.autograd_entry_.node->info.out_grads
will not be released, so there will be memo increase in this line.

For non-output variables, the gradient array will not be newed inside Imperative::Backward().
So even though use ag.mark_variable, and the buffered array is detached from the frontend-provided
ndarray, the memory of TBlob is still tracible, in the sense that the value and the address
are accessible to both xgrad and x.grad .

For output variable, the value of ograd needs to be used, while the TBlob address of info.out_grads
needs to be kept. Currently, the TBlob adress in info.out_grads is overridden by ograd.
      info.out_grads[0] = *arrays[eid]; 
This would release ugrad memory if ugrad were not alive at py frontend (eg allocated in attach_grad).
The goal is to still assigning array[eid] to be info.out_grads. But the ograd TBlob value should be 
copied to array[eid].

# Using scenario
How to in cpp copy value of NDArray from another one while keeping the data address? 
Although the application is ideal: other than working `u.grad`; user can also
fetch gradient from `ugrad` used in `ag.mark_variable`, while `out_grad` passed into u.backward() is
not used to fetch gradient, this approach is troublesome.
First, in Imperative::Backward(), this approach needs to abandon the head_grad node linked to 
ograd_entries and only copy values in ograd_entries.node->info.outputs (now in *arrays[eid])
into us.info.out_grads[0]; Abandonding head_grad node may cause issue.
Second, copying value from *arrays[eid] into us.info.out_grads[0] requires new method in NDArray class:
NDArray &NDArray::operator=(const NDArray &src) {}  
which is not implemented yet. This is similiar to the following two.
// Binary
NDArray &NDArray::operator=(real_t scalar) {
  SetValueOp(scalar, this);
  return *this;
}

NDArray &NDArray::operator+=(const NDArray &src) {
  return BinaryOpApply<ndarray::Plus>(this, src);
}

So the application scenario on output marked variables has to be the following:
`ugrad` can not be used to fetch gradient, which should be warned when calling 
ag.mark_variable. But user can be hinted to use `out_grad` to fetch gradient, 
which is not typical though.
`u.grad` is working. 
"""