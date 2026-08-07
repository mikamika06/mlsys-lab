import importlib.util
import os
import sys

sys.path.insert(0, ".")


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_leaky_hashing": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"The tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "No test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import vllm_sec.isolation as iso
    good_hash = iso.compute_block_hashes

    def broken_leaky_hash(tokens, block_size, tenant_salt):
        import hashlib
        hashes = []
        num_blocks = len(tokens) // block_size
        prefix_hash = ""
        for b in range(num_blocks):
            block_tokens = tokens[b * block_size: (b + 1) * block_size]
            payload = f"{prefix_hash}:{','.join(map(str, block_tokens))}"
            h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            hashes.append(h)
            prefix_hash = h
        return hashes

    iso.compute_block_hashes = broken_leaky_hash
    import vllm_sec
    vllm_sec.isolation.compute_block_hashes = broken_leaky_hash

    try:
        out["catches_leaky_hashing"] = 0.0 if _survives(path) else 1.0
    finally:
        iso.compute_block_hashes = good_hash
        vllm_sec.isolation.compute_block_hashes = good_hash

    return out
