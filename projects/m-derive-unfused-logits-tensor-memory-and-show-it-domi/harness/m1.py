import ref


def check(workdir):
    from logits.memory import unfused_logits_bytes, weight_memory_bytes, logits_dominates_weights

    out = {"memory_ratio_match": 0.0}
    ok = 0
    for cfg in ref.CONFIGS:
        bs = cfg["batch_size"]
        sl = cfg["seq_len"]
        vs = cfg["vocab_size"]
        np_ = int(cfg["num_params"])

        want_logits = ref.unfused_logits_bytes(bs, sl, vs)
        got_logits = unfused_logits_bytes(bs, sl, vs)

        want_weights = ref.weight_memory_bytes(np_)
        got_weights = weight_memory_bytes(np_)

        want_dom = ref.logits_dominates_weights(bs, sl, vs, np_)
        got_dom = logits_dominates_weights(bs, sl, vs, np_)

        if got_logits == want_logits and got_weights == want_weights and got_dom == want_dom:
            ok += 1

    if ok == len(ref.CONFIGS):
        out["memory_ratio_match"] = 1.0
    return out
