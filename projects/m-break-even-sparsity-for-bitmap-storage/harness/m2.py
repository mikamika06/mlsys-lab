import ref
import numpy as np


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
        
    out = {"rel_err": 1.0, "break_even_ok": 0.0}
    try:
        from edge_prune.storage import find_break_even_sparsity, calculate_theoretical_size
        
        be_8 = find_break_even_sparsity(8)
        be_16 = find_break_even_sparsity(16)
        if abs(be_8 - 0.125) < 1e-5 and abs(be_16 - 0.0625) < 1e-5:
            out["break_even_ok"] = 1.0
            
        masks = ref.get_global_masks(ref.FIXTURES, 0.5)
        want_8 = ref.calculate_theoretical_size(masks, 8)
        got_8 = calculate_theoretical_size(masks, 8)
        
        want_16 = ref.calculate_theoretical_size(masks, 16)
        got_16 = calculate_theoretical_size(masks, 16)
        
        diff = abs(want_8 - got_8) + abs(want_16 - got_16)
        total = want_8 + want_16
        out["rel_err"] = float(diff / total) if total > 0 else 1.0
        
    except Exception as e:
        out["_note"] = str(e)
        
    return out
