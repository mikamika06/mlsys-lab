import ref


def check(workdir):
    from pipelib.formulas import f1b_bubble_ratio, gpipe_bubble_ratio, interleaved_1f1b_bubble_ratio

    out = {"rel_err": 1.0}
    max_err = 0.0

    for cfg in ref.CONFIGS:
        want = ref.eval_formulas(cfg)
        p, m, v = cfg["p"], cfg["m"], cfg["v"]

        try:
            got_gpipe = gpipe_bubble_ratio(p, m)
            got_f1b = f1b_bubble_ratio(p, m)
            got_interleaved = interleaved_1f1b_bubble_ratio(p, m, v)
        except Exception as e:
            out["_note"] = f"Formula execution raised exception: {e}"
            return out

        err_gpipe = abs(got_gpipe - want["gpipe"]) / max(1e-9, abs(want["gpipe"]))
        err_f1b = abs(got_f1b - want["f1b"]) / max(1e-9, abs(want["f1b"]))
        err_interleaved = abs(got_interleaved - want["interleaved"]) / max(1e-9, abs(want["interleaved"]))

        max_err = max(max_err, err_gpipe, err_f1b, err_interleaved)

    out["rel_err"] = float(max_err)
    return out
