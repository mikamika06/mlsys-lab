import ref

def check(workdir):
    from fp16pred.ranking import rank_sensitivity, generate_golden
    out = {"rankings_match": 0.0, "golden_match": 0.0}

    want_ranking = ref.rank_sensitivity(ref.CONFIGS)
    got_ranking = rank_sensitivity(ref.CONFIGS)
    if got_ranking == want_ranking:
        out["rankings_match"] = 1.0
    else:
        out["_note"] = f"ranking mismatch: got {got_ranking}, want {want_ranking}"
        return out

    want_golden = ref.generate_golden(ref.CONFIGS)
    got_golden = generate_golden(ref.CONFIGS)
    if got_golden == want_golden:
        out["golden_match"] = 1.0
    else:
        out["_note"] = f"golden mismatch: got {got_golden}, want {want_golden}"

    return out
