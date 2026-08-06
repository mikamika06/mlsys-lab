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


def extract_tensor_bytes(shape, dtype_str):
    if dtype_str not in DTYPE_SIZES:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    return num_elements * DTYPE_SIZES[dtype_str]


def parse_do_bench_trace(trace_data):
    samples = trace_data.get("samples_ms", [])
    if not samples:
        return {"mean_ms": 0.0, "std_ms": 0.0}
    arr = np.array(samples, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
    }
