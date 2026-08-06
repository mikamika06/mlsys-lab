import sys
import numpy as np

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        import tp_sp.forward as fwd
    except Exception as e:
        return {"ag_ok": 0.0, "rs_ok": 0.0, "fw_tp_ok": 0.0, "fw_sp_ok": 0.0, "sp_matches_tp": 0.0, "_note": str(e)}
    
    out = {"ag_ok": 0.0, "rs_ok": 0.0, "fw_tp_ok": 0.0, "fw_sp_ok": 0.0, "sp_matches_tp": 0.0}
    
    S, B, H, H_inner = 16, 2, 8, 16
    tp = 4

    try:
        ag_in = [np.ones((S//tp, B, H)) * i for i in range(tp)]
        ag_out = fwd.all_gather(ag_in)
        if len(ag_out) == tp and ag_out[0].shape == (S, B, H):
            if np.allclose(ag_out[0][:S//tp], 0) and np.allclose(ag_out[0][-S//tp:], 3):
                out["ag_ok"] = 1.0
    except Exception:
        pass

    try:
        rs_in = [np.ones((S, B, H)) * (i+1) for i in range(tp)]
        rs_out = fwd.reduce_scatter(rs_in)
        if len(rs_out) == tp and rs_out[0].shape == (S//tp, B, H):
            if np.allclose(rs_out[0], sum(range(1, tp+1))):
                out["rs_ok"] = 1.0
    except Exception:
        pass
        
    try:
        np.random.seed(42)
        X = np.random.randn(S, B, H)
        X_sharded = list(np.split(X, tp, axis=0))
        X_repl = [X.copy() for _ in range(tp)]

        W1 = np.random.randn(H, H_inner)
        W1_list = list(np.split(W1, tp, axis=1))

        W2 = np.random.randn(H_inner, H)
        W2_list = list(np.split(W2, tp, axis=0))

        out_tp = fwd.forward_tp(X_repl, W1_list, W2_list)
        if len(out_tp) == tp and out_tp[0].shape == (S, B, H):
            out["fw_tp_ok"] = 1.0

        out_sp = fwd.forward_sp(X_sharded, W1_list, W2_list)
        if len(out_sp) == tp and out_sp[0].shape == (S//tp, B, H):
            out["fw_sp_ok"] = 1.0

        if out["fw_tp_ok"] == 1.0 and out["fw_sp_ok"] == 1.0:
            sp_concat = np.concatenate(out_sp, axis=0)
            if np.allclose(sp_concat, out_tp[0]):
                out["sp_matches_tp"] = 1.0
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"forward crash: {e}"

    return out
