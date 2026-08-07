import math
import random
import tracemalloc


def grade(sol, fx) -> dict:
    func = getattr(sol, 'flash_attention_forward', fx)

    # Test parameters
    N, d = 256, 16
    block_size = 32

    # Deterministic random inputs as plain Python lists
    rng = random.Random(42)
    Q = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(N)]
    K = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(N)]
    V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(N)]

    # Warmup call
    try:
        _ = func(Q[:32], K[:32], V[:32], block_size=16)
    except Exception as e:
        raise AssertionError(f"TASK_FAIL: execution failed during warmup | {e}")

    # Compute dense reference oracle using pure Python
    scale = 1.0 / math.sqrt(d)
    S = []
    for i in range(N):
        row = []
        for j in range(N):
            dot = sum(Q[i][k] * K[j][k] for k in range(d))
            row.append(dot * scale)
        S.append(row)

    P = []
    for i in range(N):
        row = S[i]
        m = max(row)
        exps = [math.exp(val - m) for val in row]
        s = sum(exps)
        P.append([e / s for e in exps])

    d_v = d
    ref_out = []
    for i in range(N):
        out_row = []
        for v in range(d_v):
            val = sum(P[i][j] * V[j][v] for j in range(N))
            out_row.append(val)
        ref_out.append(out_row)

    # Measure student memory and execution
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        student_out = func(Q, K, V, block_size=block_size)
    except Exception as e:
        tracemalloc.stop()
        raise AssertionError(f"TASK_FAIL: execution failed during call | {e}")

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Compute max absolute error
    max_abs_err = 0.0
    if not isinstance(student_out, list) or len(student_out) != len(ref_out):
        max_abs_err = float('inf')
    else:
        for i in range(N):
            for j in range(d_v):
                err = abs(student_out[i][j] - ref_out[i][j])
                if err > max_abs_err:
                    max_abs_err = err

    # Compute peak alloc ratio against an N x N float64 matrix byte size
    matrix_bytes = N * N * 8
    peak_alloc_ratio = peak / matrix_bytes if matrix_bytes > 0 else 0.0

    return {
        'max_abs_err': max_abs_err,
        'peak_alloc_ratio': peak_alloc_ratio
    }
