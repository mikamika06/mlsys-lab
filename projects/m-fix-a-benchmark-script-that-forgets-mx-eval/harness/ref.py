import mlx.core as mx


def run_benchmark(x, weights):
    out = x
    for w in weights:
        out = mx.matmul(out, w)
    mx.eval(out)
    return out


def analyze_implicit_evals(operations):
    triggers = []
    for i, op in enumerate(operations):
        if op in ("item", "tolist", "numpy", "print", "bool", "float", "int"):
            triggers.append(i)
    return triggers


def benchmark_matmul_chain(x, weights, compiled=False):
    def chain(inp, ws):
        res = inp
        for w in ws:
            res = mx.matmul(res, w)
        return res

    fn = mx.compile(chain) if compiled else chain
    _ = fn(x, weights)
    mx.eval(_)
    import time
    start = time.perf_counter()
    res = fn(x, weights)
    mx.eval(res)
    end = time.perf_counter()
    return end - start
