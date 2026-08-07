import ref


def check(workdir):
    try:
        from sdpa.dispatch import predict_backend
        from sdpa.repair import repair_config_for_flash
    except ImportError:
        return {"matches": 0.0}

    ok = 0
    total = 0

    dims = [8, 16, 20, 32, 100, 128, 256, 512]
    dtypes = ["float16", "bfloat16", "float32", "float64"]
    masks = [True, False]

    for dim in dims:
        for dt in dtypes:
            for m in masks:
                total += 1
                want = ref.predict_backend(dim, dt, m)
                try:
                    got = predict_backend(dim, dt, m)
                    if got == want:
                        ok += 1
                except Exception:
                    pass

    configs = [
        {"head_dim": 20, "dtype": "float32", "has_custom_mask": True, "is_causal": False},
        {"head_dim": 128, "dtype": "float16", "has_custom_mask": False, "is_causal": True},
        {"head_dim": 300, "dtype": "bfloat16", "has_custom_mask": True, "is_causal": False},
        {"head_dim": 60, "dtype": "float64", "has_custom_mask": False, "is_causal": False},
    ]

    total += len(configs)
    for c in configs:
        want = ref.repair_config_for_flash(c)
        try:
            got = repair_config_for_flash(c)
            if got == want:
                ok += 1
        except Exception:
            pass

    return {"matches": float(ok) / total if total > 0 else 0.0}
