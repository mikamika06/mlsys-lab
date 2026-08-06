"""Reference data generator and oracle for profiling tests."""

import random

def generate_profile_pair(seed):
    rng = random.Random(seed)

    def make_report(num_kernels, duration_ns):
        trace_start = 1_000_000
        trace_end = trace_start + duration_ns
        kernels = []
        for _ in range(num_kernels):
            s = rng.randint(trace_start, trace_end - 1000)
            length = rng.randint(500, 5000)
            e = min(s + length, trace_end)
            st = rng.randint(1, 4)
            kernels.append({"start_ns": s, "end_ns": e, "stream_id": st, "device_id": 0})
        return {"kernels": kernels, "trace_start_ns": trace_start, "trace_end_ns": trace_end}

    rep_a = make_report(rng.randint(20, 50), 100_000)
    rep_b = make_report(rng.randint(5, 15), 100_000)
    return rep_a, rep_b


def generate_churn_report(seed):
    rng = random.Random(seed)
    names = ["cudaLaunchKernel", "cudaMemcpyAsync", "cudaMalloc", "cudaFree", "cudaStreamSynchronize"]
    records = []
    for name in names:
        records.append({"name": name, "total_time_ns": rng.randint(10000, 500000), "calls": rng.randint(1, 100)})
    return {"records": records}
