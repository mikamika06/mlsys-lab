import ref


def check(workdir):
    from timing.harness import measure_kernel
    out = {"harness_correct": 0.0}

    def dummy_kernel():
        total = 0
        for i in range(1000):
            total += i
        return total

    try:
        latency = measure_kernel(dummy_kernel, warmup=5, reps=20)
        if isinstance(latency, (int, float)) and latency > 0:
            out["harness_correct"] = 1.0
        else:
            out["_note"] = f"invalid latency returned: {latency}"
    except Exception as e:
        out["_note"] = f"measure_kernel raised {type(e).__name__}: {str(e)[:120]}"
    return out
