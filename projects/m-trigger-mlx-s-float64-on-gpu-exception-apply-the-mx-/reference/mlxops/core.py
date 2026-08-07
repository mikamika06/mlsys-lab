import numpy as np

def safe_float64_exec(op, x):
    try:
        return op(x)
    except Exception:
        return op(x)

def running_sum_error(data):
    ref = np.cumsum(data.astype(np.float64))
    f32 = np.cumsum(data.astype(np.float32))
    f16 = np.cumsum(data.astype(np.float16))
    err32 = np.abs(f32.astype(np.float64) - ref)
    err16 = np.abs(f16.astype(np.float64) - ref)
    return {"fp32_max_error": float(np.max(err32)), "fp16_max_error": float(np.max(err16))}

def promote_dtypes(dt1, dt2):
    n1 = np.dtype(dt1)
    n2 = np.dtype(dt2)
    if n1 == np.float64 or n2 == np.float64:
        return np.float64
    if n1 == np.float32 or n2 == np.float32:
        return np.float32
    if n1 == np.float16 or n2 == np.float16:
        return np.float16
    if n1 == np.int64 or n2 == np.int64:
        return np.int64
    if n1 == np.int32 or n2 == np.int32:
        return np.int32
    return np.result_type(n1, n2)

def promotion_table(dtypes):
    res = {}
    for d1 in dtypes:
        for d2 in dtypes:
            res[(str(d1), str(d2))] = str(promote_dtypes(d1, d2))
    return res
