import ref


def check(workdir):
    from benchkit import parse, report

    rows = parse.load_all(ref.files())
    want = ref.raw()
    out = {"split_match": 0.0, "ubatch_choice": 0.0, "bandwidth_order": 0.0,
           "floor_respected": 0.0}

    s = report.prefill_decode_split(rows)
    pre = [r for r in want if r.get("n_prompt", 0) and not r.get("n_gen", 0)]
    dec = [r for r in want if r.get("n_gen", 0) and not r.get("n_prompt", 0)]
    best_pre = max(ref.expect_derive(r)["tokens_per_second"] for r in pre)
    best_dec = max(ref.expect_derive(r)["tokens_per_second"] for r in dec)
    if (s.get("prefill_rows") == len(pre) and s.get("decode_rows") == len(dec)
            and ref.near(s.get("best_prefill_ts", -1), best_pre, 1e-9)
            and ref.near(s.get("best_decode_ts", -1), best_dec, 1e-9)):
        out["split_match"] = 1.0

    pick = report.pick_ubatch(rows, min_decode_ts=0.0)
    by_ub = {}
    for r in pre:
        by_ub.setdefault(int(r.get("n_ubatch", 0)), []).append(
            ref.expect_derive(r)["tokens_per_second"])
    best_ub = max(by_ub, key=lambda u: max(by_ub[u])) if by_ub else None
    if pick.get("chosen") == best_ub:
        out["ubatch_choice"] = 1.0

    high = report.pick_ubatch(rows, min_decode_ts=1e9)
    if high.get("chosen") is not None and high.get("options"):
        if all(not o["meets_floor"] for o in high["options"]):
            out["floor_respected"] = 1.0

    summary = report.model_summary(rows)
    if len(summary) >= 2:
        ranked = sorted(summary.values(), key=lambda v: -v["bytes_per_second_decode"])
        if "moe" in ranked[0]["model"].lower():
            out["bandwidth_order"] = 1.0
    return out
