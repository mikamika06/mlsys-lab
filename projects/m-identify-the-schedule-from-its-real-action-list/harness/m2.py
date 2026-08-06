import ref


def check(workdir):
    from pipesched.metrics import compute_bubble_fraction

    out = {"bubble_rel_err": 1.0}
    ratios = [0.05, 0.12, 0.25, 0.40]
    errors = []
    for r in ratios:
        logs = ref.generate_mock_logs(r, seed=100)
        want = ref.compute_bubble_fraction(logs)
        try:
            got = compute_bubble_fraction(logs)
        except Exception:
            got = -1.0
        err = abs(got - want) / (abs(want) + 1e-9)
        errors.append(err)
    max_err = max(errors) if errors else 1.0
    out["bubble_rel_err"] = float(max_err)
    if max_err > 0.01:
        out["_note"] = f"max relative error {max_err:.4f} exceeds 0.01 threshold"
    return out
