import time
import mlx.core as mx


def benchmark_matmul_chain(x, weights, compiled=False):
    def chain(inp, ws):
        res = inp
        for w in ws:
            res = mx.matmul(res, w)
        return res

    fn = mx.compile(chain) if compiled else chain

    _ = fn(x, weights)
    mx.eval(_)

    start = time.perf_counter()
    res = fn(x, weights)
    mx.eval(res)
    end = time.perf_counter()
    return end - start
