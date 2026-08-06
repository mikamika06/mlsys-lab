import ref

def check(workdir):
    from specadapt.tuning import estimate_alpha
    alphas = ref.generate_drifting_stream(seed=123, steps=50)
    total_err = 0.0
    history = []
    for a in alphas:
        history.append(a)
        if len(history) > 5:
            history.pop(0)
        want = ref.ref_estimate_alpha(history)
        try:
            got = estimate_alpha(history)
        except Exception:
            got = -1.0
        total_err += abs(want - got)
    mean_err = total_err / len(alphas)
    return {"alpha_error": float(mean_err)}
