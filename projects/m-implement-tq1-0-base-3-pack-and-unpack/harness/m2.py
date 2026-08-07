import ref
from ternary.analysis import compute_iq4_xs_size, compute_q4_k_s_size, measure_imatrix_effect


def check(workdir):
    out = {"size_parity_match": 0.0, "codebook_valid": 0.0}
    try:
        sz_iq = compute_iq4_xs_size(4096)
        sz_qk = compute_q4_k_s_size(4096)
        if sz_iq > 0 and sz_qk > 0:
            out["size_parity_match"] = 1.0
        cb, iw = ref.get_codebook_params()
        res = measure_imatrix_effect(cb, iw)
        if isinstance(res, (int, float)) and res >= 0.0:
            out["codebook_valid"] = 1.0
    except Exception as e:
        out["_note"] = f"m2 execution error: {type(e).__name__}: {str(e)[:100]}"
    return out
