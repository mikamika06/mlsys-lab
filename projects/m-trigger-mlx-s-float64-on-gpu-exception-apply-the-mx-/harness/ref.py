import numpy as np

class StreamContext:
    _current = "gpu"

    def __init__(self, device):
        self.device = str(device).lower()
        self._prev = None

    def __enter__(self):
        self._prev = StreamContext._current
        StreamContext._current = self.device
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        StreamContext._current = self._prev

def get_active_device():
    return StreamContext._current

def execute_op(op, tensor):
    if get_active_device() == "gpu" and getattr(tensor, "dtype", None) == np.float64:
        raise ValueError("float64 is not supported on GPU")
    return op(tensor)

def safe_float64_exec(op, tensor):
    if getattr(tensor, "dtype", None) == np.float64 and get_active_device() == "gpu":
        with StreamContext("cpu"):
            return execute_op(op, tensor)
    return execute_op(op, tensor)

RNG = np.random.default_rng(1337)
DATASET = RNG.uniform(0.01, 0.5, size=2000)

def measure_running_sum_error(data):
    d64 = np.asarray(data, dtype=np.float64)
    ref = np.cumsum(d64)

    f32 = np.cumsum(d64.astype(np.float32)).astype(np.float64)
    f16 = np.cumsum(d64.astype(np.float16)).astype(np.float64)

    err32 = np.abs(f32 - ref)
    err16 = np.abs(f16 - ref)

    max32 = float(np.max(err32))
    max16 = float(np.max(err16))

    ratio = float(err16[-1] / (err32[-1] + 1e-12))

    return {
        "max_err_fp32": max32,
        "max_err_fp16": max16,
        "final_err_fp32": float(err32[-1]),
        "final_err_fp16": float(err16[-1]),
        "drift_ratio": ratio,
    }

PROMOTION_DTYPES = ["bool", "int32", "int64", "float16", "bfloat16", "float32", "float64"]

_FLOAT_RANKS = {"float64": 4, "float32": 3, "float16": 2, "bfloat16": 2}
_INT_RANKS = {"int64": 2, "int32": 1}

def _norm_dtype(dt):
    s = str(dt).lower().replace("dtype('", "").replace("')", "").strip()
    if s in ("float", "float_"):
        return "float64"
    if s in ("int", "int_"):
        return "int64"
    return s

def promote_dtypes(dt1, dt2):
    d1 = _norm_dtype(dt1)
    d2 = _norm_dtype(dt2)

    if d1 == d2:
        return d1
    if d1 == "bool":
        return d2
    if d2 == "bool":
        return d1

    is_f1 = d1 in _FLOAT_RANKS
    is_f2 = d2 in _FLOAT_RANKS
    is_i1 = d1 in _INT_RANKS
    is_i2 = d2 in _INT_RANKS

    if is_f1 and is_f2:
        if {d1, d2} == {"float16", "bfloat16"}:
            return "float32"
        r1 = _FLOAT_RANKS[d1]
        r2 = _FLOAT_RANKS[d2]
        return d1 if r1 >= r2 else d2

    if is_f1 and is_i2:
        return d1
    if is_f2 and is_i1:
        return d2

    if is_i1 and is_i2:
        r1 = _INT_RANKS[d1]
        r2 = _INT_RANKS[d2]
        return d1 if r1 >= r2 else d2

    return d1

def compute_promotion_table(dtypes):
    table = {}
    for d1 in dtypes:
        for d2 in dtypes:
            table[(str(d1), str(d2))] = promote_dtypes(d1, d2)
    return table
