import ref

def check(workdir):
    from streamkv.selection import streaming_llm_indices

    out = {"sinks_matched": 0.0}
    ok = 0
    for case in ref.CASES:
        want = ref.streaming_llm_indices(case["seq_len"], case["num_sinks"], case["window_size"])
        got = streaming_llm_indices(case["seq_len"], case["num_sinks"], case["window_size"])
        if sorted(got) == sorted(want):
            ok += 1
    out["sinks_matched"] = float(ok)
    return out
