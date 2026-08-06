import ref


def check(workdir):
    from logits.chunked import chunked_crossentropy_bytes, memory_savings_ratio

    out = {"savings_match": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        bs = cfg["batch_size"]
        sl = cfg["seq_len"]
        vs = cfg["vocab_size"]
        cs = cfg["chunk_size"]

        want_bytes = ref.chunked_crossentropy_bytes(bs, sl, vs, cs)
        got_bytes = chunked_crossentropy_bytes(bs, sl, vs, cs)

        want_ratio = ref.memory_savings_ratio(bs, sl, vs, cs)
        got_ratio = memory_savings_ratio(bs, sl, vs, cs)

        if got_bytes == want_bytes and abs(got_ratio - want_ratio) < 1e-5:
            ok += 1

    if ok == len(ref.CONFIGS):
        out["savings_match"] = 1.0
    return out
