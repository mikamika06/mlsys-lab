import ref


def check(workdir):
    from cachekey.oracle import can_share_blocks, infer_prefix_residency

    out = {"oracles_matched": 0.0, "sharing_matched": 0.0}

    try:
        want_res = ref.infer_prefix_residency(
            ref.TTFT_BENCHMARKS["samples"],
            ref.TTFT_BENCHMARKS["baseline_ttft_per_token"],
            ref.TTFT_BENCHMARKS["cache_hit_ttft"],
        )
        got_res = infer_prefix_residency(
            ref.TTFT_BENCHMARKS["samples"],
            ref.TTFT_BENCHMARKS["baseline_ttft_per_token"],
            ref.TTFT_BENCHMARKS["cache_hit_ttft"],
        )
        if got_res == want_res:
            out["oracles_matched"] = 1.0
        else:
            out["_note"] = f"oracle mismatch: got {got_res}, reference {want_res}"
    except Exception as e:
        out["_note"] = f"infer_prefix_residency raised {type(e).__name__}: {e}"
        return out

    sharing_ok = True
    for i, (req_a, req_b, allow_ct) in enumerate(ref.PAIR_TEST_CASES):
        want_share = ref.can_share_blocks(req_a, req_b, allow_cross_tenant=allow_ct)
        try:
            got_share = can_share_blocks(req_a, req_b, allow_cross_tenant=allow_ct)
            if got_share != want_share:
                sharing_ok = False
                out["_note"] = f"pair case {i}: got {got_share}, reference {want_share}"
                break
        except Exception as e:
            sharing_ok = False
            out["_note"] = f"can_share_blocks case {i} raised {type(e).__name__}: {e}"
            break

    if sharing_ok:
        out["sharing_matched"] = 1.0

    return out
