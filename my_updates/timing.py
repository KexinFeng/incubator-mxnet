import multiprocessing
import time

def mxnet_worker():
    b_time = time.time()
    print('{}'.format(b_time))
    # import mxnet
    print ('time consumes: {}'.format(time.time()-b_time))
    a = 1
    b = 2
    print(a+b)

read_process = [multiprocessing.Process(target=mxnet_worker) for i in range(8)]
for p in read_process:
    p.daemon = True
    p.start()