import ref

def check(workdir):
    from kvtransfer.breakeven import compute_breakeven_prompt_len
    from kvtransfer.sizing import size_prefill_decode_ratio
    out = {"breakeven_match": 0.0, "sizing_match": 0.0}
    try:
        for cfg in ref.CONFIGS:
            for bw in ref.BANDWIDTHS_GBPS:
                want_be = ref.compute_breakeven_prompt_len(cfg, bw)
                got_be = compute_breakeven_prompt_len(cfg, bw)
                if got_be != want_be:
                    out["_note"] = f"breakeven mismatch at {bw} Gbps: got {got_be}, want {want_be}"
                    return out
        out["breakeven_match"] = 1.0

        for req in ref.REQUESTS:
            p_len = req["prompt_len"]
            o_len = req["output_len"]
            want_sz = ref.compute_sizing_ratio(p_len, o_len, 0.5, 20.0)
            got_sz = size_prefill_decode_ratio(p_len, o_len, 0.5, 20.0)
            if abs(got_sz - want_sz) > 1e-6:
                out["_note"] = f"sizing ratio mismatch: got {got_sz}, want {want_sz}"
                return out
        out["sizing_match"] = 1.0
    except Exception as e:
        out["_note"] = f"Exception: {type(e).__name__}: {str(e)[:120]}"
    return out
