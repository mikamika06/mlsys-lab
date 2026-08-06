import ref

def check(workdir):
    from sharder.vocab import export_vocab_only
    from sharder.schedule import compute_conversion_schedule

    out = {"vocab_matched": 0.0, "splits_matched": 0.0}

    cfg = ref.MODELS[0]
    want_vocab = ref.convert_vocab_only(cfg["vocab"])
    got_vocab = export_vocab_only(cfg["vocab"])

    if got_vocab == want_vocab:
        out["vocab_matched"] = 1.0
    else:
        out["_note"] = f"vocab mismatch: got {got_vocab}, want {want_vocab}"
        return out

    want_shards = ref.plan_shards(cfg)
    got_sched = compute_conversion_schedule(cfg["vocab"], cfg["tensors"], cfg["max_bytes"])

    if got_sched.get("shards") == want_shards and got_sched.get("vocab") == want_vocab:
        out["splits_matched"] = 1.0
    else:
        out["_note"] = f"schedule mismatch: got {got_sched}"

    return out
