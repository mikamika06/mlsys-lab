import numpy as np
import ref


def check(workdir):
    from awq.search import find_best_alpha
    
    out = {"grid_search_matched": 0.0, "rel_err": 1.0}
    X, W = ref.generate_synthetic_data(seed=202)
    alphas = np.linspace(0.0, 1.0, 11)
    
    ref_alpha, ref_scales, ref_mse = ref.ref_find_best_alpha(X, W, alphas)
    try:
        got_alpha, got_scales, got_mse = find_best_alpha(X, W, alphas)
        
        rel_mse_err = abs(got_mse - ref_mse) / max(ref_mse, 1e-12)
        out["rel_err"] = float(rel_mse_err)
        
        alpha_match = abs(got_alpha - ref_alpha) < 1e-6
        scales_match = np.allclose(got_scales, ref_scales, rtol=1e-5, atol=1e-5)
        
        if alpha_match and scales_match and rel_mse_err <= 1e-5:
            out["grid_search_matched"] = 1.0
        else:
            out["_note"] = f"Mismatch: got alpha={got_alpha}, ref alpha={ref_alpha}, rel_err={rel_mse_err}"
    except Exception as e:
        out["_note"] = f"find_best_alpha raised exception: {type(e).__name__}: {e}"
        
    return out
