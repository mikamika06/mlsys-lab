import ref


def check(workdir):
    from fsdpfit.derive import derive_world_size

    out = {"world_size_matched": 0.0}
    ok = True
    scenarios = [
        (1000, 100, 600),
        (5000, 200, 2000),
        (100, 50, 100),
    ]
    for model_b, ov_b, budget in scenarios:
        want = ref.derive_world_size(model_b, ov_b, budget)
        got = derive_world_size(model_b, ov_b, budget)
        if want != got:
            ok = False
            out["_note"] = f"model={model_b}, overhead={ov_b}, budget={budget}: got {got}, want {want}"
            break
    if ok:
        out["world_size_matched"] = 1.0
    return out
