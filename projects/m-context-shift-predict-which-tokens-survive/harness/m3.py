import importlib.util
import os

def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return e
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
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
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_reuse": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    res = _run(path)
    if isinstance(res, Exception):
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(res).__name__}: {str(res)[:120]}"
        return out
    if res is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import caching.blocks as b
    good = b.surviving_blocks

    def bad_surviving_blocks(new_prompt, cached_seqs, block_contents):
        best_blocks = []
        for seq in cached_seqs:
            current_blocks = []
            token_offset = 0
            for block_id in seq:
                block_tokens = block_contents[block_id]
                b_len = len(block_tokens)
                if token_offset + b_len > len(new_prompt):
                    break
                if new_prompt[token_offset : token_offset + b_len] == block_tokens:
                    current_blocks.append(block_id)
                token_offset += b_len
            if len(current_blocks) > len(best_blocks):
                best_blocks = current_blocks
        return best_blocks

    b.surviving_blocks = bad_surviving_blocks

    try:
        if not _survives(path):
            out["catches_bad_reuse"] = 1.0
        else:
            out["_note"] = "Test did not fail when disjoint blocks were incorrectly matched."
    finally:
        b.surviving_blocks = good

    return out
