import ref

def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    try:
        from prefill.memory import cheapest_config
    except ImportError:
        return {"mem_matched": 0.0}

    out = {"mem_matched": 0.0}
    ok = 0

    for model, gpus, ctx, budget in ref.MEMORY_FIXTURES:
        want = ref.cheapest_config(model, gpus, ctx, budget)
        try:
            got = cheapest_config(model, gpus, ctx, budget)
            if want == got:
                ok += 1
            else:
                out["_note"] = f"want {want}, got {got}"
        except Exception as e:
            out["_note"] = f"exception: {e}"

    out["mem_matched"] = float(ok)
    return out
