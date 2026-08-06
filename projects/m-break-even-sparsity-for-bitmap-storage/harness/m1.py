import ref
import numpy as np


def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    
    out = {"rel_err": 1.0, "layer_ok": 0.0, "global_ok": 0.0}
    try:
        from edge_prune.pruning import get_layer_masks, get_global_masks
        
        sparsities = [0.1, 0.5, 0.9]
        layer_diff = 0.0
        global_diff = 0.0
        total_mag = 0.0
        
        for s in sparsities:
            want_l = ref.get_layer_masks(ref.FIXTURES, s)
            got_l = get_layer_masks(ref.FIXTURES, s)
            
            for k, tensor in ref.FIXTURES.items():
                w_mask = want_l[k]
                g_mask = got_l[k]
                w_mag = float(np.sum(np.abs(tensor[w_mask])))
                g_mag = float(np.sum(np.abs(tensor[g_mask])))
                layer_diff += abs(w_mag - g_mag)
                total_mag += w_mag
                
            want_g = ref.get_global_masks(ref.FIXTURES, s)
            got_g = get_global_masks(ref.FIXTURES, s)
            
            for k, tensor in ref.FIXTURES.items():
                w_mask = want_g[k]
                g_mask = got_g[k]
                w_mag = float(np.sum(np.abs(tensor[w_mask])))
                g_mag = float(np.sum(np.abs(tensor[g_mask])))
                global_diff += abs(w_mag - g_mag)
                total_mag += w_mag
                
        out["rel_err"] = float((layer_diff + global_diff) / total_mag) if total_mag > 0 else 1.0
        if layer_diff < 1e-5:
            out["layer_ok"] = 1.0
        if global_diff < 1e-5:
            out["global_ok"] = 1.0
            
    except Exception as e:
        out["_note"] = str(e)
        
    return out
