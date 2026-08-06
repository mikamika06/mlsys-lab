import numpy as np

DTYPE_SIZES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int32": 4,
    "int64": 8,
}


def generate_test_records():
    shapes = [[512, 512], [1024, 1024], [2048, 2048], [4096, 4096], [8192, 1024]]
    dtypes = ["float16", "float32", "float16", "float32", "bfloat16"]
    records = []
    np.random.seed(42)

    for i in range(5):
        base_ms = (i + 1) * 0.5
        unfused_samples = (
            base_ms * 2.2 + np.random.normal(0, 0.02, 10)
        ).tolist()
        fused_samples = (base_ms + np.random.normal(0, 0.01, 10)).tolist()

        records.append(
            {
                "shape": shapes[i],
                "dtype": dtypes[i],
                "num_inputs": 2,
                "num_outputs": 1,
                "num_unfused_ops": 3,
                "unfused_trace": {"samples_ms": unfused_samples},
                "fused_trace": {"samples_ms": fused_samples},
            }
        )
    return records


def reference_parse_trace(trace):
    samples = trace.get("samples_ms", [])
    if not samples:
        return 0.0, 0.0
    arr = np.array(samples, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def reference_summary(record):
    u_mean, _ = reference_parse_trace(record["unfused_trace"])
    f_mean, _ = reference_parse_trace(record["fused_trace"])

    num_elems = 1
    for d in record["shape"]:
        num_elems *= d
    bytes_per_elem = DTYPE_SIZES[record["dtype"]]
    elem_bytes = num_elems * bytes_per_elem

    num_in = record.get("num_inputs", 2)
    num_out = record.get("num_outputs", 1)
    num_ops = record.get("num_unfused_ops", 2)

    unfused_bytes = elem_bytes * (num_in + num_out + (num_ops - 1))
    fused_bytes = elem_bytes * (num_in + num_out)

    speedup = u_mean / f_mean if f_mean > 0 else 0.0
    time_saved = u_mean - f_mean

    u_gbps = (unfused_bytes / 1e9) / (u_mean / 1000.0) if u_mean > 0 else 0.0
    f_gbps = (fused_bytes / 1e9) / (f_mean / 1000.0) if f_mean > 0 else 0.0

    return {
        "unfused_mean_ms": u_mean,
        "fused_mean_ms": f_mean,
        "speedup": speedup,
        "time_saved_ms": time_saved,
        "unfused_gbps": u_gbps,
        "fused_gbps": f_gbps,
        "throughput_ratio": f_gbps / u_gbps if u_gbps > 0 else 0.0,
    }
