from contextlib import contextmanager
from time import time

@contextmanager
def timeit(prefix):
    t = time()
    yield
    print("{}: {} seconds".format(prefix, time() - t))

with timeit("import mxnet"):
    import mxnet