import ref


def check(workdir):
    from coreprune.palettize import chain_prune_palettize
    models = ref.get_test_models()
    ok = 0
    out = {"combined_reduction_matched": 0.0, "models": float(len(models))}
    for i, w in enumerate(models):
        _, got_size = chain_prune_palettize(w, 0.5, 4)
        _, want_size = ref.oracle_chain(w, 0.5, 4)
        _, prune_size = ref.oracle_prune(w, 0.5)
        if got_size < prune_size and abs(got_size - want_size) <= max(1, int(w.nbytes * 0.05)):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: chain got size {got_size}, want ~{want_size}, prune size {prune_size}"
    out["combined_reduction_matched"] = 1.0 if ok == len(models) else 0.0
    return out
