import ref

def check(workdir):
    from prune_net.cnn_pruner import prune_real_cnn
    profile = ref.get_model_profile()
    ratio = 0.5
    got = prune_real_cnn(profile, ratio)
    want_len = len(profile["layers"])
    got_len = len(got.get("pruned_layers", []))
    ok = 1.0 if got_len == want_len else 0.0
    out = {"cnn_pruned_correctly": ok}
    if ok != 1.0:
        out["_note"] = f"expected {want_len} layers, got {got_len}"
    return out
