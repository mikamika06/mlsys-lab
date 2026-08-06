import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        from divergence.analyze import compute_divergences
    except ImportError:
        return {"byte_exact_fraction": 0.0}

    a, b = ref.get_fixtures_m1()

    try:
        got = compute_divergences(a, b)
    except Exception:
        return {"byte_exact_fraction": 0.0}

    want = ref.compute_divergences(a, b)
    if got == want:
        return {"byte_exact_fraction": 1.0}

    return {"byte_exact_fraction": 0.0}
