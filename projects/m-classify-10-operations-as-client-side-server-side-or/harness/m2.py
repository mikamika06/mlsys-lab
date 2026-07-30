import ref


def check(workdir):
    from opside import hop_count, is_legal_pipeline, side_sequence

    out = {"sequence_match": 1.0, "hops_match": 1.0, "legality_match": 1.0}
    for i, case in enumerate(ref.PIPELINES):
        cfg, order = case["config"], case["order"]
        want_seq = ref.side_sequence(cfg, order)
        want_hops = ref.hop_count(cfg, order)
        want_legal = ref.is_legal_pipeline(cfg, order)

        got_seq = list(side_sequence(cfg, order))
        got_hops = hop_count(cfg, order)
        got_legal = bool(is_legal_pipeline(cfg, order))

        if got_seq != want_seq:
            out["sequence_match"] = 0.0
            if "_note" not in out:
                out["_note"] = f"pipeline {i}: sequence got {got_seq}, reference {want_seq}"
        if got_hops != want_hops:
            out["hops_match"] = 0.0
            if "_note" not in out:
                out["_note"] = f"pipeline {i}: hop_count got {got_hops}, reference {want_hops}"
        if got_legal != want_legal:
            out["legality_match"] = 0.0
            if "_note" not in out:
                out["_note"] = f"pipeline {i}: is_legal_pipeline got {got_legal}, reference {want_legal}"
    return out
