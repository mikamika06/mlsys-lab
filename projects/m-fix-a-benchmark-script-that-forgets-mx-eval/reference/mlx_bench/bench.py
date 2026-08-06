import mlx.core as mx


def run_benchmark(x, weights):
    out = x
    for w in weights:
        out = mx.matmul(out, w)
    mx.eval(out)
    return out
