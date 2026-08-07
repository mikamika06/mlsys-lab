import ref

def check(workdir):
    from prune.eval import compare_methods

    m = {"compare_ok": 0.0, "wanda_better": 0.0}
    w, x = ref.get_fixture()

    mse_mag, mse_wan = compare_methods(w, x, 0.5)
    if isinstance(mse_mag, float) and isinstance(mse_wan, float):
        m["compare_ok"] = 1.0
        if mse_wan < mse_mag:
            m["wanda_better"] = 1.0

    return m
