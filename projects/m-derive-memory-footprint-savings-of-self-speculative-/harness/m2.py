import ref


def check(workdir):
    from speculative.derive import derive_savings
    out = {"rel_err_match": 0.0, "savings_positive": 0.0}
    rel_err_ok = True
    savings_pos_ok = True
    for i, (target_cfg, draft_cfg, bs, sl) in enumerate(ref.CONFIGS):
        ref_res = ref.derive_savings(target_cfg, draft_cfg, bs, sl)
        try:
            got_res = derive_savings(target_cfg, draft_cfg, bs, sl)
            if not isinstance(got_res, dict):
                rel_err_ok = False
                continue
            for k in ["saved_bytes", "savings_ratio"]:
                ref_val = ref_res[k]
                got_val = got_res.get(k, 0.0)
                err = abs(got_val - ref_val) / (abs(ref_val) + 1e-9)
                if err > 1e-4:
                    rel_err_ok = False
                    if "_note" not in out:
                        out["_note"] = f"mismatch in {k}: got {got_val}, want {ref_val}, rel_err {err}"
            if draft_cfg.get("is_self_speculative", False):
                if got_res.get("saved_bytes", 0) <= 0:
                    savings_pos_ok = False
            else:
                if got_res.get("saved_bytes", 0) != 0:
                    savings_pos_ok = False
        except Exception as e:
            rel_err_ok = False
            savings_pos_ok = False
            if "_note" not in out:
                out["_note"] = f"derive_savings raised error: {type(e).__name__}: {str(e)[:100]}"
    out["rel_err_match"] = 1.0 if rel_err_ok else 0.0
    out["savings_positive"] = 1.0 if savings_pos_ok else 0.0
    return out
