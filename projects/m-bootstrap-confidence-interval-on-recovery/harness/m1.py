import ref

def check(workdir):
    from evalrep.evaluator import run_lm_eval

    out = {"evals_matched": 0.0}
    ok = 0
    model_path = "models/small-quantized"
    try:
        got = run_lm_eval(model_path, ref.TASKS)
        want = ref.get_reference_evals(model_path)
        for task in ref.TASKS:
            if task in got and len(got[task]) == len(want[task]):
                ok += 1
        out["evals_matched"] = float(ok)
    except Exception as e:
        out["_note"] = f"run_lm_eval failed: {type(e).__name__}: {str(e)[:120]}"
    return out
