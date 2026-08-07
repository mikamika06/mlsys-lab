import ref


def check(workdir):
    from trtopt.tune import tune_max_tokens
    out = {"opt_match": 0.0}
    candidates = [256, 512, 1024, 2048]
    slo = 400.0
    rate = 12.0
    prefill = [128, 256]
    for btype in ["static", "continuous"]:
        want = ref.tune_max_tokens(candidates, slo, btype, rate, prefill)
        got = tune_max_tokens(candidates, slo, btype, rate, prefill)
        if want != got:
            out["_note"] = f"batching {btype}: got {got}, want {want}"
            return out
    out["opt_match"] = 1.0
    return out
