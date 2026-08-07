import ref


def check(workdir):
    from sdpa_pred.eligibility import is_eligible

    out = {"eligibility_matched": 0.0}
    backends = ["flash_attention", "mem_efficient", "math"]
    ok = 0
    total = 0
    for cfg in ref.CONFIGS:
        for b in backends:
            total += 1
            want = ref.is_eligible(b, cfg["dtype"], cfg["is_causal"], cfg["q_len"], cfg["kv_len"], cfg["head_dim"], cfg["device_cap"])
            got = is_eligible(b, cfg["dtype"], cfg["is_causal"], cfg["q_len"], cfg["kv_len"], cfg["head_dim"], cfg["device_cap"])
            if want == got:
                ok += 1
    if ok == total:
        out["eligibility_matched"] = 1.0
    else:
        out["_note"] = f"matched {ok}/{total} eligibility checks"
    return out
