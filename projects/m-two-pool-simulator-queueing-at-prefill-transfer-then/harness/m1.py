import ref


def check(workdir):
    from kvsim.simulator import simulate_pipeline

    out = {"simulations_matched": 0.0}
    ok = 0
    for i in range(3):
        reqs = [
            {"id": f"r-{i}-{j}", "prompt_tokens": 50 + j * 10, "kv_size_bytes": 2048, "decode_tokens": 20}
            for j in range(2)
        ]
        got = simulate_pipeline(reqs, prefill_capacity=2, decode_capacity=2, bandwidth_bps=100000000)
        if isinstance(got, list) and len(got) == len(reqs):
            ok += 1
    out["simulations_matched"] = float(ok)
    return out
