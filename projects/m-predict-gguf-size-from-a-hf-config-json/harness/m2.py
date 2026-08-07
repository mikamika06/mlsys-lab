import ref


def check(workdir):
    from ggufsize.calculator import compute_size
    errors = []
    for cfg in ref.CONFIGS:
        for q in ["F16", "Q4_K_M"]:
            want = ref.predict_size(cfg, q)
            try:
                got = compute_size(cfg, q)
            except Exception:
                got = 0
            if want > 0:
                err = abs(got - want) / float(want)
                errors.append(err)
            else:
                errors.append(1.0)

    max_err = max(errors) if errors else 1.0
    return {"rel_err": float(max_err)}
