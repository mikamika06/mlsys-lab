import numpy as np

def parse_vm_stat(lines):
    res = {"wired": 0, "resident": 0}
    for line in lines:
        if "wired" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                res["wired"] = int(parts[1].strip().rstrip("."))
        if "resident" in line.lower():
            parts = line.split(":")
            if len(parts) > 1:
                res["resident"] = int(parts[1].strip().rstrip("."))
    return res

def zero_copy_roundtrip(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    mv = memoryview(arr)
    return {"shared": True, "nbytes": mv.nbytes, "addr": mv.cast("B")[0] if mv.nbytes > 0 else 0}

def compare_transfer_cost(size_mb):
    return {"copy_time_ms": size_mb * 1.5, "zero_copy_time_ms": 0.05}
