import mxnet as mx

json = "{\"nodes\": [{\"op\":\"null\",\"name\":\".Inputs.Input1\",\"inputs\":[]},{\"op\":\"null\",\"name\":\".Inputs.Input2\",\"inputs\":[]},{\"op\":\"elemwise_add\",\"name\":\".$0\",\"inputs\":[[0,0,0],[1,0,0]]},{\"op\":\"_copy\",\"name\":\".Outputs.Output\",\"inputs\":[[2,0,0]]}],\"arg_nodes\":[0,1],\"heads\":[[3,0,0]]}"
sym = mx.symbol.fromjson(json)
# digraph = mx.viz.plot_network(sym)
# digraph.view()
# mx.viz.print_summary(sym)

ex = sym._bind(
    mx.cpu(), 
    {'.Inputs.Input1': mx.nd.array([0.1, 0.2, 0.3, 0.4]), '.Inputs.Input2': mx.nd.array([0.5, 0.6, 0.7, 0.8])},
    args_grad={
        '.Inputs.Input1': mx.ndarray.zeros((1)), 
        '.Inputs.Input2': mx.ndarray.zeros((1))
    },
    grad_req={'.Inputs.Input1': 'null', '.Inputs.Input2': 'write'}
                                # null -> not attach grad
)
ex.forward(is_train=True)
print(ex.outputs)
ex.backward(out_grads=mx.nd.array([7.0, 6.0, 5.0, 4.0]))
print(ex.grad_arrays)

