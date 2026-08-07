import numpy as np
import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from paged_fa.utils import reconstruct_contiguous
        from paged_fa.attention import standard_attention
    except ImportError:
        return {"reconstruct_max_abs_err": 100.0, "standard_attn_max_abs_err": 100.0, "_note": "Import failed"}

    q, k_cache, v_cache, block_tables, context_lens = ref.generate_fixtures(42)

    out = {"reconstruct_max_abs_err": 100.0, "standard_attn_max_abs_err": 100.0}
    try:
        k_contig, v_contig = reconstruct_contiguous(k_cache, v_cache, block_tables, context_lens)
        ref_k, ref_v = ref.reconstruct_contiguous(k_cache, v_cache, block_tables, context_lens)

        err_k = np.max(np.abs(k_contig - ref_k))
        err_v = np.max(np.abs(v_contig - ref_v))
        out["reconstruct_max_abs_err"] = float(max(err_k, err_v))

        out_attn = standard_attention(q, ref_k, ref_v, context_lens)
        ref_out = ref.standard_attention(q, ref_k, ref_v, context_lens)
        out["standard_attn_max_abs_err"] = float(np.max(np.abs(out_attn - ref_out)))
    except Exception as e:
        out["_note"] = f"Failed with {e}"

    return out
