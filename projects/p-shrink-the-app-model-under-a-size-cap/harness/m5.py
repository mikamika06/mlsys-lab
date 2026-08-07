import ref


def check(workdir):
    import compress.api as api

    m = {"compresses": 0.0, "under_limit": 0.0, "accuracy_ok": 0.0}
    model = ref.get_model()

    try:
        comp = api.compress_model(model)
        m["compresses"] = 1.0

        true_total = ref.get_sizes(comp)
        if true_total <= 40_000_000:
            m["under_limit"] = 1.0

        acc = ref.evaluate(model, comp)
        if acc >= 84.0:
            m["accuracy_ok"] = 1.0
    except Exception:
        pass

    return m
