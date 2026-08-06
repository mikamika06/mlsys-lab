import ref

def check(workdir):
    from partitioner.tune import optimize_allowlist
    
    out = {"rel_err": 0.0, "cands_matched": 0.0}
    errs = []
    cands_ok = 0
    
    for i, cfg in enumerate(ref.CONFIGS):
        want_cand, want_cost = ref.optimize_allowlist(**cfg)
        got_cand, got_cost = optimize_allowlist(**cfg)
        
        if want_cand == got_cand:
            cands_ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got cand {got_cand}, reference cand {want_cand}"
            
        if want_cost > 0:
            err = abs(float(got_cost) - float(want_cost)) / float(want_cost)
        else:
            err = 0.0 if float(got_cost) == float(want_cost) else 1.0
        errs.append(err)
        
    out["rel_err"] = sum(errs) / len(errs)
    out["cands_matched"] = float(cands_ok)
    return out
