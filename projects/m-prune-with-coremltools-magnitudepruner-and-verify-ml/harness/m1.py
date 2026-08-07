import ref


def check(workdir):
    from coreprune.prune import prune_weights
    models = ref.get_test_models()
    ok = 0
    out = {"pruned_ratio_matched": 0.0, "models": float(len(models))}
    for i, w in enumerate(models):
        _, got_size = prune_weights(w, 0.5)
        _, want_size = ref.oracle_prune(w, 0.5)
        orig_size = w.nbytes
        if got_size < orig_size and abs(got_size - want_size) <= max(1, int(orig_size * 0.05)):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"model {i}: got size {got_size}, want ~{want_size}, orig {orig_size}"
    out["pruned_ratio_matched"] = 1.0 if ok == len(models) else 0.0
    return out
