import ref

def check(workdir):
    from kvtransfer.transfer import compute_kv_bytes, compute_transfer_times
    out = {"bytes_and_time_match": 0.0}
    try:
        for cfg in ref.CONFIGS:
            for req in ref.REQUESTS:
                p_len = req["prompt_len"]
                want_bytes = ref.compute_kv_bytes(cfg, p_len)
                got_bytes = compute_kv_bytes(cfg, p_len)
                if got_bytes != want_bytes:
                    out["_note"] = f"kv_bytes mismatch: got {got_bytes}, want {want_bytes}"
                    return out

                want_times = ref.compute_transfer_times(want_bytes, ref.BANDWIDTHS_GBPS)
                got_times = compute_transfer_times(got_bytes, ref.BANDWIDTHS_GBPS)
                for bw in ref.BANDWIDTHS_GBPS:
                    if abs(got_times.get(bw, -1) - want_times[bw]) > 1e-9:
                        out["_note"] = f"transfer time mismatch at {bw} Gbps: got {got_times.get(bw)}, want {want_times[bw]}"
                        return out
        out["bytes_and_time_match"] = 1.0
    except Exception as e:
        out["_note"] = f"Exception: {type(e).__name__}: {str(e)[:120]}"
    return out
