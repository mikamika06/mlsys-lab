import ref

def check(workdir):
    from speculative_quant.cutoff import find_int8_cutoff

    out = {"cutoffs_matched": 0.0}
    ok = 0
    for i, sc in enumerate(ref.SCENARIOS):
        want = ref.find_int8_cutoff(
            sc["draft_sizes"], sc["s_target"], sc["K"], sc["mem_bw"],
            sc["alphas_fp16"], sc["alphas_int8"], sc["overheads"]
        )
        got = find_int8_cutoff(
            sc["draft_sizes"], sc["s_target"], sc["K"], sc["mem_bw"],
            sc["alphas_fp16"], sc["alphas_int8"], sc["overheads"]
        )
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"scenario {i}: got {got}, want {want}"

    out["cutoffs_matched"] = float(ok) / len(ref.SCENARIOS)
    return out
