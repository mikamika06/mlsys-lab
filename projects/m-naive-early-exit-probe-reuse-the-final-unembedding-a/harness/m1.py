import ref


def check(workdir):
    from earlyexit.probe import compute_early_exit_agreement
    h, w, f, _, _ = ref.generate_fixture()
    want = ref.compute_early_exit_agreement(h, w, f)
    try:
        got = compute_early_exit_agreement(h, w, f)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"raised {type(e).__name__}: {e}"}

    rel_err = float(abs(got - want) / (abs(want) + 1e-9))
    return {"rel_err": rel_err}
