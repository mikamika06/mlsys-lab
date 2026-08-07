import importlib.util
import os
import sys
import numpy as np

def _run(path, workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod)
           if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True

def _survives(path, workdir):
    try:
        return _run(path, workdir) is True
    except Exception:
        return False

def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_broken_len": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path, workdir)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import paged_fa.attention as p_attn
    good_paged = p_attn.paged_attention

    def broken_paged_attention(q, k_cache, v_cache, block_tables, context_lens):
        batch_size, num_heads, head_dim = q.shape
        num_blocks, block_size, _, _ = k_cache.shape
        out_mat = np.zeros_like(q)
        scale = 1.0 / np.sqrt(head_dim)
        for b in range(batch_size):
            req_blocks = (context_lens[b] + block_size - 1) // block_size
            seq_len_bug = req_blocks * block_size
            block_table = block_tables[b]
            for h in range(num_heads):
                q_vec = q[b, h]
                scores = []
                for i in range(seq_len_bug):
                    logical_block = i // block_size
                    offset = i % block_size
                    physical_block = block_table[logical_block]
                    k_vec = k_cache[physical_block, offset, h, :]
                    scores.append(np.dot(k_vec, q_vec) * scale)
                scores = np.array(scores)
                scores = scores - np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                out_vec = np.zeros(head_dim)
                for i in range(seq_len_bug):
                    logical_block = i // block_size
                    offset = i % block_size
                    physical_block = block_table[logical_block]
                    v_vec = v_cache[physical_block, offset, h, :]
                    out_vec += probs[i] * v_vec
                out_mat[b, h] = out_vec
        return out_mat

    p_attn.paged_attention = broken_paged_attention
    try:
        survived = _survives(path, workdir)
        out["catches_broken_len"] = 0.0 if survived else 1.0
    finally:
        p_attn.paged_attention = good_paged

    return out
