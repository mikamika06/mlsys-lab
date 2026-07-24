import dis
import time
import zlib
import numpy as np


def _python_cpu_loop(n=10000):
    x = 0
    for i in range(n):
        x += i
    return x


def _numpy_blas(a):
    return np.dot(a, a)


def _blocking_io():
    time.sleep(0.001)


def _c_extension_compress(data):
    return zlib.compress(data)


def _oracle(fn):
    instructions = list(dis.get_instructions(fn))
    names = set(fn.__code__.co_names)

    has_python_loop = any(i.opname in {"FOR_ITER", "JUMP_BACKWARD"} for i in instructions)
    has_python_arithmetic = any(i.opname == "BINARY_OP" for i in instructions)

    if has_python_loop and has_python_arithmetic:
        return False

    if "sleep" in names:
        return True

    if "compress" in names:
        return True

    if "dot" in names:
        sample = np.ones((4, 4), dtype=np.float64)
        out = _numpy_blas(sample)
        return isinstance(out, np.ndarray)

    return False


def grade(sol, fx) -> dict:
    workloads = [
        {"name": "python_loop", "fn": _python_cpu_loop},
        {"name": "numpy_blas", "fn": _numpy_blas},
        {"name": "blocking_io", "fn": _blocking_io},
        {"name": "c_extension", "fn": _c_extension_compress},
    ]

    expected = {w["name"]: _oracle(w["fn"]) for w in workloads}

    try:
        got = sol.classify_thread_scaling(workloads)
    except Exception:
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == expected else 0.0}
