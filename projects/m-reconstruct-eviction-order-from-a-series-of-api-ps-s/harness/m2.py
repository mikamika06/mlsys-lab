import ref


def check(workdir):
    from evict.unload import force_unload_and_verify

    before = [{"id": "model_a", "status": "loaded"}, {"id": "model_b", "status": "loaded"}]
    after = ref.simulate_unload("model_a", before)
    proven = force_unload_and_verify("model_a", before, after)
    out = {"unload_proven": 1.0 if proven else 0.0}
    if not proven:
        out["_note"] = "force_unload_and_verify did not correctly prove model_a removal"
    return out
