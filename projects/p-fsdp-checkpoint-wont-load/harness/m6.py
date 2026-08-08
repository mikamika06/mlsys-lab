import importlib.util
import os


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
    path = os.path.join(workdir, "tests", "test_regression.py")
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_padding": 0.0}

    if not os.path.isfile(path):
        return out

    import fsdp_ckpt.converter as converter

    try:
        first = _run(path)
    except Exception:
        out["has_tests"] = 1.0
        return out

    if first is None:
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    good_shard = converter.shard_checkpoint

    def bad_shard(consolidated, num_ranks):
        import numpy as np
        ranks = [{} for _ in range(num_ranks)]
        for k, tensor in consolidated.items():
            flat = tensor.flatten()
            chunk_size = len(flat) // num_ranks
            flat = flat[:chunk_size * num_ranks]
            for i in range(num_ranks):
                ranks[i][k] = flat[i * chunk_size : (i + 1) * chunk_size]
        return ranks

    converter.shard_checkpoint = bad_shard
    try:
        if not _survives(path):
            out["catches_broken_padding"] = 1.0
    finally:
        converter.shard_checkpoint = good_shard

    return out
