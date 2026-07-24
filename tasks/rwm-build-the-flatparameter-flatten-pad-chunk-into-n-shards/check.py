import numpy as np

from mlsys import scorers


def _oracle(params, world_size):
    flat = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in params])
    total = flat.size
    remainder = total % world_size
    pad = 0 if remainder == 0 else (world_size - remainder)
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float64)])
    shard_size = flat.size // world_size
    shards = [flat[i * shard_size:(i + 1) * shard_size] for i in range(world_size)]
    return shards, shard_size


def _build_cases():
    rng = np.random.default_rng(0)
    cases = []
    # case 0: total not divisible by world_size -> needs padding
    params = [rng.normal(size=(3, 5)), rng.normal(size=(7,)), rng.normal(size=(2, 2))]
    cases.append((params, 4))
    # case 1: total exactly divisible -> zero padding
    params = [rng.normal(size=(4, 4)), rng.normal(size=(8,))]
    cases.append((params, 3))
    # case 2: single param, world_size 1
    params = [rng.normal(size=(10,))]
    cases.append((params, 1))
    # case 3: many small params, larger world_size
    params = [rng.normal(size=(2,)) for _ in range(9)]
    cases.append((params, 5))
    return cases


def grade(sol, fx) -> dict:
    size_ok = 1.0
    byte_frac = 1.0

    for params, world_size in _build_cases():
        ref_shards, shard_size = _oracle(params, world_size)

        try:
            got = sol.flatten_pad_shard([p.copy() for p in params], world_size)
        except Exception:
            return {"size_ok": 0.0, "byte_exact_fraction": 0.0}

        try:
            got = [np.asarray(s, dtype=np.float64) for s in got]
        except Exception:
            return {"size_ok": 0.0, "byte_exact_fraction": 0.0}

        if len(got) != world_size or any(s.shape != (shard_size,) for s in got):
            size_ok = 0.0
            byte_frac = 0.0
            continue

        ref_bytes = np.concatenate(ref_shards)
        got_bytes = np.concatenate(got)
        frac = scorers.byte_exact_fraction(ref_bytes, got_bytes)
        byte_frac = min(byte_frac, frac)

    return {"size_ok": size_ok, "byte_exact_fraction": byte_frac}
