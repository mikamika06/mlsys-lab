import ref


def check(workdir):
    from streamkv.analysis import compute_kv_bytes, compute_sink_window_bytes

    cases = ref.get_test_cases()
    matched = 0
    for c in cases:
        want_full = ref.compute_kv_bytes(c["seq_len"], c["layers"], c["kv_heads"], c["head_dim"])
        want_sink = ref.compute_sink_window_bytes(c["seq_len"], c["layers"], c["kv_heads"], c["head_dim"], c["sink"], c["window"])

        got_full = compute_kv_bytes(c["seq_len"], c["layers"], c["kv_heads"], c["head_dim"])
        got_sink = compute_sink_window_bytes(c["seq_len"], c["layers"], c["kv_heads"], c["head_dim"], c["sink"], c["window"])

        if got_full == want_full and got_sink == want_sink:
            matched += 1

    return {"memory_bytes_matched": float(matched)}
