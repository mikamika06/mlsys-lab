import ref


def check(workdir):
    from eval.hellaswag import run_hellaswag_eval

    out = {"evaluations_matched": 0.0}
    ok = 0
    for items in ref.HELLASWAG_TASKS:
        want = ref.run_hellaswag_eval(items, ref.dummy_model_fn)
        got = run_hellaswag_eval(items, ref.dummy_model_fn)
        if (
            abs(got.get("acc", 0.0) - want["acc"]) < 1e-6
            and abs(got.get("stderr", 0.0) - want["stderr"]) < 1e-6
            and got.get("count") == want["count"]
        ):
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"got {got}, expected {want}"

    out["evaluations_matched"] = float(ok)
    return out
