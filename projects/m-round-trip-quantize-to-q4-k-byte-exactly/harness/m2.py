import ref
import numpy as np


def check(workdir):
    out = {"subblock_mse_match": 0.0}
    try:
        from q4k.quant import quantize_q4_k_superblock
        from q4k.analysis import find_worst_subblocks, compare_q4k_q40_mse
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out
    w = ref.generate_superblock()
    data = quantize_q4_k_superblock(w)
    try:
        ref_ranking = ref.find_worst_subblocks(w, data)
        got_ranking = find_worst_subblocks(w, data)
        ref_cmp = ref.compare_q4k_q40_mse(w)
        got_cmp = compare_q4k_q40_mse(w)
        if got_ranking == ref_ranking and abs(got_cmp.get("ratio", 0) - ref_cmp.get("ratio", 0)) < 1e-4:
            out["subblock_mse_match"] = 1.0
        else:
            out["_note"] = f"ranking or MSE metrics mismatch: got {got_ranking[:3]} vs ref {ref_ranking[:3]}"
    except Exception as e:
        out["_note"] = f"execution error: {e}"
    return out
