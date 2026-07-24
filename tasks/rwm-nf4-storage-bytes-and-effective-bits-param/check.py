import numpy as np


def _oracle(n, blocksize, scale_dtype):
    itemsize = np.dtype(scale_dtype).itemsize
    n_blocks = n // blocksize
    codes_bytes = n // 2
    scale_bytes = n_blocks * itemsize
    total_bytes = codes_bytes + scale_bytes
    bits_per_param = 8.0 * total_bytes / n
    return int(total_bytes), float(bits_per_param)


def grade(sol, fx) -> dict:
    """
    Random (n, blocksize, scale_dtype) with n an exact multiple of both 2
    and blocksize; compares (total_bytes, bits_per_param) against the
    closed-form NF4 storage formula.
    """
    rng = np.random.default_rng(0)
    dtypes = ["float32", "float16", "int8", "uint8"]
    blocksizes = [16, 32, 64, 128, 256, 512]

    ok = 1.0
    for _ in range(8):
        blocksize = int(rng.choice(blocksizes))
        k = int(rng.integers(1, 40))
        n = blocksize * k * 2  # divisible by blocksize and by 2
        scale_dtype = str(rng.choice(dtypes))

        expected_bytes, expected_bits = _oracle(n, blocksize, scale_dtype)
        try:
            got = sol.nf4_storage(n, blocksize, scale_dtype)
        except Exception:
            ok = 0.0
            break

        if not isinstance(got, tuple) or len(got) != 2:
            ok = 0.0
            break

        got_bytes, got_bits = got
        try:
            same_bytes = int(got_bytes) == expected_bytes
            same_bits = abs(float(got_bits) - expected_bits) < 1e-9
        except Exception:
            ok = 0.0
            break

        if not (same_bytes and same_bits):
            ok = 0.0
            break

    return {"exact_match": ok}
