import ref


def check(workdir):
    from prefix_cache.layout import optimize_prompt_layout
    from prefix_cache.lookup import compute_trace_hit_rate

    out = {"hit_rate_match": 0.0, "layout_optim_match": 0.0}

    hit_rate_ok = True
    for trace in ref.TEST_TRACES:
        for bsize in [2, 4]:
            want_hr = ref.compute_trace_hit_rate(trace, bsize)
            try:
                got_hr = compute_trace_hit_rate(trace, bsize)
            except Exception as e:
                out["_note"] = f"compute_trace_hit_rate raised: {e}"
                return out

            if abs(want_hr - got_hr) > 1e-6:
                hit_rate_ok = False
                out["_note"] = f"Hit rate mismatch: got {got_hr}, want {want_hr}"
                break
        if not hit_rate_ok:
            break

    if hit_rate_ok:
        out["hit_rate_match"] = 1.0

    layout_ok = True
    for comp_set in ref.TEST_COMPONENTS:
        for bsize in [4, 8]:
            want = ref.optimize_prompt_layout(comp_set, bsize)
            try:
                got = optimize_prompt_layout(comp_set, bsize)
            except Exception as e:
                out["_note"] = f"optimize_prompt_layout raised: {e}"
                return out

            if (
                got.get("optimized_components") != want["optimized_components"]
                or got.get("prompt_tokens") != want["prompt_tokens"]
                or got.get("prefix_block_tokens") != want["prefix_block_tokens"]
            ):
                layout_ok = False
                out["_note"] = (
                    f"Layout optimizer output mismatch for block size {bsize}"
                )
                break
        if not layout_ok:
            break

    if layout_ok:
        out["layout_optim_match"] = 1.0

    return out
