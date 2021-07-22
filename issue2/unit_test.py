import mxnet as mx

json = "{\"nodes\": [{\"op\":\"null\",\"name\":\".Inputs.Input1\",\"inputs\":[]},{\"op\":\"null\",\"name\":\".Inputs.Input2\",\"inputs\":[]},{\"op\":\"elemwise_add\",\"name\":\".$0\",\"inputs\":[[0,0,0],[1,0,0]]},{\"op\":\"_copy\",\"name\":\".Outputs.Output\",\"inputs\":[[2,0,0]]}],\"arg_nodes\":[0,1],\"heads\":[[3,0,0]]}"
# json = "{\"nodes\": [{\"op\":\"null\",\"name\":\".Inputs.Input1\",\"inputs\":[]},{\"op\":\"null\",\"name\":\".Inputs.Input2\",\"inputs\":[]},{\"op\":\"elemwise_mul\",\"name\":\".$0\",\"inputs\":[[0,0,0],[1,0,0]]},{\"op\":\"_copy\",\"name\":\".Outputs.Output\",\"inputs\":[[2,0,0]]}],\"arg_nodes\":[0,1],\"heads\":[[3,0,0]]}"
sym = mx.symbol.fromjson(json)
# digraph = mx.viz.plot_network(sym)
# digraph.view()
# mx.viz.print_summary(sym)

ex = sym._bind(
    mx.cpu(), 
    {'.Inputs.Input1': mx.nd.array([0.4]), '.Inputs.Input2': mx.nd.array([0.5])},
    args_grad={
        '.Inputs.Input1': mx.ndarray.zeros((1)), 
        '.Inputs.Input2': mx.ndarray.zeros((1))
    },
    grad_req={'.Inputs.Input1': 'null', '.Inputs.Input2': 'write'}
                                # null -> not attach grad
)
ex.forward(is_train=True)
print(ex.outputs)
ex.backward(out_grads=mx.nd.array([20.21]))
print(ex.grad_arrays)


def test_gradient():
    x = mx.nd.ones((1,))
    x.attach_grad()

    with mx.autograd.record():
        z = mx.nd.elemwise_add(mx.nd.exp(x), x)
    dx, = mx.autograd.grad(z, [x], create_graph=True)
    assert abs(dx.asscalar() - 3.71828175) < 1e-7
    dx.backward()
    assert abs(x.grad.asscalar() - 2.71828175) < 1e-7


