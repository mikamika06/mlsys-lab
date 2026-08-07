import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_non_chained_hash": 0.0,
    }
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"The tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "No test_* functions found in tests/test_regression.py"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import prefix_cache.hash as hash_mod

    good_chain = hash_mod.build_prefix_hash_chain

    def broken_non_chained(token_ids, block_size):
        num_blocks = len(token_ids) // block_size
        hashes = []
        for i in range(num_blocks):
            block_tokens = token_ids[i * block_size : (i + 1) * block_size]
            curr_hash = hash_mod.compute_block_hash(
                block_tokens, parent_hash=None
            )
            hashes.append(curr_hash)
        return hashes

    hash_mod.build_prefix_hash_chain = broken_non_chained
    import prefix_cache

    prefix_cache.hash.build_prefix_hash_chain = broken_non_chained

    try:
        out["catches_non_chained_hash"] = 0.0 if _survives(path) else 1.0
    finally:
        hash_mod.build_prefix_hash_chain = good_chain
        prefix_cache.hash.build_prefix_hash_chain = good_chain

    return out
