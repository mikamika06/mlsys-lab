"""Milestone 2 harness check."""
import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"diagnosis_matches": 0.0, "prep_matches": 0.0}
    try:
        from tpquant.checker import diagnose_config
        from tpquant.parallel import prepare_tp_linear
    except ImportError as e:
        out["_note"] = f"ImportError: {e}"
        return out

    diag_ok = 0
    total = len(ref.CONFIGS)
    for cfg in ref.CONFIGS:
        ref_diag = ref.diagnose_config(cfg["in_features"], cfg["group_size"], cfg["tp_size"], cfg["desc_act"])
        got_diag = diagnose_config(cfg["in_features"], cfg["group_size"], cfg["tp_size"], cfg["desc_act"])
        if got_diag == ref_diag:
            diag_ok += 1
        elif "_note" not in out:
            out["_note"] = f"diag mismatch on cfg {cfg}: got {got_diag}, want {ref_diag}"

    if diag_ok == total:
        out["diagnosis_matches"] = 1.0

    prep_ok = 0
    rng = np.random.RandomState(123)
    for cfg in ref.CONFIGS:
        in_f = cfg["in_features"]
        out_f = cfg["out_features"]
        g_s = cfg["group_size"]
        tp_s = cfg["tp_size"]
        perm = cfg["perm"]
        total_g = in_f // g_s
        scales = rng.randn(total_g, out_f).astype(np.float32)

        mode = "replicate_scales" if cfg["desc_act"] and tp_s > 1 else "validate_only"
        try:
            ref_res = ref.prepare_tp_linear(in_f, out_f, g_s, tp_s, perm, scales, mode)
            got_res = prepare_tp_linear(in_f, out_f, g_s, tp_s, perm, scales, mode)
            if got_res["mode"] == ref_res["mode"] and len(got_res["ranks"]) == len(ref_res["ranks"]):
                match = True
                for r_got, r_ref in zip(got_res["ranks"], ref_res["ranks"]):
                    if not (
                        r_got["rank"] == r_ref["rank"]
                        and np.allclose(r_got["scales"], r_ref["scales"])
                        and (r_got["g_idx"] == r_ref["g_idx"]).all()
                    ):
                        match = False
                        break
                if match:
                    prep_ok += 1
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"prep exception on cfg {cfg}: {e}"

    if prep_ok == total:
        out["prep_matches"] = 1.0

    return out
