import ref


def check(workdir):
    out = {"triggers_identified": 0.0}
    try:
        from mlx_bench.graph import analyze_implicit_evals
    except Exception as e:
        out["_note"] = f"import failed: {e}"
        return out

    ops = ["matmul", "item", "add", "numpy", "tolist", "print"]
    got = analyze_implicit_evals(ops)
    want = ref.analyze_implicit_evals(ops)

    if got == want:
        out["triggers_identified"] = float(len(got))
    else:
        out["_note"] = f"got {got}, want {want}"
    return out
