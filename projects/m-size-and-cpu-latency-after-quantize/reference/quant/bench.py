import numpy as np


def calculate_model_bytes(model_dict):
    total = 0
    for key, info in model_dict.items():
        if isinstance(info, np.ndarray):
            total += info.nbytes
        elif isinstance(info, dict):
            if "bytes" in info:
                total += info["bytes"]
            else:
                shape = info.get("shape", ())
                nelem = 1
                for d in shape:
                    nelem *= d
                dtype = info.get("dtype", "float32")
                if dtype == "float32":
                    b = nelem * 4
                elif dtype in ("float16", "bfloat16"):
                    b = nelem * 2
                elif dtype == "int8":
                    b = nelem * 1
                elif dtype == "int4":
                    b = int(np.ceil(nelem / 2.0))
                else:
                    b = nelem * 4
                b += info.get("extra_bytes", 0)
                total += b
    return total


def profile_quantization(before_model, after_model, bench_fn):
    lat_before = float(bench_fn(before_model))
    lat_after = float(bench_fn(after_model))
    bytes_before = calculate_model_bytes(before_model)
    bytes_after = calculate_model_bytes(after_model)
    ratio = float(bytes_after / bytes_before) if bytes_before > 0 else 1.0
    speedup = float(lat_before / lat_after) if lat_after > 0 else 1.0
    return {
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "size_ratio": ratio,
        "latency_before_ms": lat_before,
        "latency_after_ms": lat_after,
        "speedup": speedup,
    }
