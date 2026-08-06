import numpy as np
import ref


def check(workdir):
    from awq.scale import fold_scales
    
    out = {"scale_folding_exact": 0.0}
    X, W = ref.generate_synthetic_data(seed=101)
    scales = np.random.uniform(0.1, 10.0, size=X.shape[1])
    
    Y_ref = X @ W
    try:
        X_s, W_s = fold_scales(X, W, scales)
        Y_folded = X_s @ W_s
        err = np.max(np.abs(Y_ref - Y_folded))
        if err < 1e-10:
            out["scale_folding_exact"] = 1.0
        else:
            out["_note"] = f"Scale folding non-exact, max diff: {err}"
    except Exception as e:
        out["_note"] = f"fold_scales raised exception: {type(e).__name__}: {e}"
        
    return out
