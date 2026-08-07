import numpy as np
import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from paged_fa.attention import paged_attention
    except ImportError:
        return {"paged_vs_ref_max_abs_err": 100.0, "_note": "Import failed"}

    q, k_cache, v_cache, block_tables, context_lens = ref.generate_fixtures(123)

    out = {"paged_vs_ref_max_abs_err": 100.0}
    try:
        out_paged = paged_attention(q, k_cache, v_cache, block_tables, context_lens)
        ref_out = ref.paged_attention(q, k_cache, v_cache, block_tables, context_lens)
        out["paged_vs_ref_max_abs_err"] = float(np.max(np.abs(out_paged - ref_out)))
    except Exception as e:
        out["_note"] = f"Failed with {e}"

    return out
