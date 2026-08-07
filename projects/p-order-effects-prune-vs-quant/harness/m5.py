import ref

def check(workdir):
    try:
        from compression.pipeline import justify_best_order
    except ImportError:
        return {"m5_ok": 0.0}

    w = ref.generate_fixture()
    try:
        res = justify_best_order(w, 0.5, 4)
    except Exception:
        return {"m5_ok": 0.0}

    if res.get("best_method") == "joint" and res.get("improvement_over_pq", 0) > 0.0:
        return {"m5_ok": 1.0}
    return {"m5_ok": 0.0}
