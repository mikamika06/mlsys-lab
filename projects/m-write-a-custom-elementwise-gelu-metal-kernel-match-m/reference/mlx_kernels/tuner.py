import time
import mlx.core as mx
from mlx_kernels.matmul import matmul_metal


def tune_threadgroup_size(a: mx.array, b: mx.array, candidate_shapes: list) -> tuple:
    best_shape = None
    best_latency = float("inf")

    for shape in candidate_shapes:
        try:
            out = matmul_metal(a, b, threadgroup_shape=shape)
            mx.eval(out)

            start = time.perf_counter()
            for _ in range(5):
                res = matmul_metal(a, b, threadgroup_shape=shape)
                mx.eval(res)
            end = time.perf_counter()

            avg_latency = (end - start) / 5.0
            if avg_latency < best_latency:
                best_latency = avg_latency
                best_shape = shape
        except Exception:
            continue

    return best_shape if best_shape is not None else candidate_shapes[0]
