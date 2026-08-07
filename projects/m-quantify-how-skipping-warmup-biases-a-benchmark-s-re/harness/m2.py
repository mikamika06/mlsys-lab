import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from bench.measure import compare_engines
    except ImportError as e:
        return {"_note": f"ImportError: {e}", "configs_matched": 0.0}

    lens = [1, 10, 100, 500]
    ref_mlx = ref.MockEngine("mlx")
    ref_llama = ref.MockEngine("llama")
    want = ref.compare_engines(ref_mlx, ref_llama, lens, 1, 4, "fp16")

    stu_mlx = ref.MockEngine("mlx")
    stu_llama = ref.MockEngine("llama")
    try:
        got = compare_engines(stu_mlx, stu_llama, lens, 1, 4, "fp16")
    except Exception as e:
        return {"_note": f"Error running compare_engines: {e}", "configs_matched": 0.0}

    if len(got) != len(want):
        return {"_note": f"Expected {len(want)} results, got {len(got)}", "configs_matched": 0.0}

    out = {}
    matched = 0
    for w, g in zip(want, got):
        try:
            if (w["len"] == g["len"] and
                abs(w["mlx"] - g["mlx"]) < 1e-5 and
                abs(w["llama"] - g["llama"]) < 1e-5 and
                w["mlx_slower"] == g["mlx_slower"]):
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Mismatch: want {w}, got {g}"
        except KeyError:
            if "_note" not in out:
                out["_note"] = "Missing key in output"

    out["configs_matched"] = float(matched)
    return out
