import random
from mlsys import scorers

def zero_copy_slice_ref(buf, start, stop):
    """Reference: the intended solution."""
    return memoryview(buf)[start:stop]

def grade(sol, fx) -> dict:
    rng = random.Random(42)
    all_ok = True
    for _ in range(5):
        n = rng.randint(50, 300)
        data = bytearray(rng.randint(0, 255) for _ in range(n))
        start = rng.randint(0, n - 2)
        stop = rng.randint(start + 1, n)
        try:
            mv = sol.zero_copy_slice(data, start, stop)
        except Exception:
            all_ok = False
            break
        # Type check: must be memoryview
        if not isinstance(mv, memoryview):
            all_ok = False
            break
        # Content check: bytes equal
        expected_bytes = data[start:stop]
        if bytes(mv) != expected_bytes:
            all_ok = False
            break
        # Zero-copy check: .obj must be the original buffer
        if mv.obj is not data:
            all_ok = False
            break
    return {"byte_exact_fraction": 1.0 if all_ok else 0.0}
