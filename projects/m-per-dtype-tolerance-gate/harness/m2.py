import numpy as np
import ref


def check(workdir):
    from tolgate.bisection import bisect_divergence

    out = {"bisection_accuracy": 0.0}

    def s0(x):
        return x + 1.0

    def s1(x):
        return x * 2.0

    def s2_eager(x):
        return x - 0.5

    def s2_compiled(x):
        return x - 50.0

    def s3(x):
        return x ** 2

    eager_pipeline = [s0, s1, s2_eager, s3]
    compiled_pipeline = [s0, s1, s2_compiled, s3]
    init_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    want_idx = ref.bisect_divergence(
        eager_pipeline, compiled_pipeline, init_val, "float32", 100
    )
    got_idx = bisect_divergence(
        eager_pipeline, compiled_pipeline, init_val, "float32", 100
    )

    if want_idx == got_idx == 2:
        out["bisection_accuracy"] = 1.0

    return out
