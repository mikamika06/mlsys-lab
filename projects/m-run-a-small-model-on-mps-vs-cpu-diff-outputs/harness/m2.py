import ref

def check(workdir):
    from mpscheck.cost import measure_staging_cost
    sizes = [64, 128, 256]
    want = ref.measure_staging_cost(sizes)
    try:
        got = measure_staging_cost(sizes)
    except Exception as e:
        return {"cost_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    if isinstance(got, dict) and set(got.keys()) == set(want.keys()):
        return {"cost_matched": 1.0}
    return {"cost_matched": 0.0, "_note": f"keys mismatch: got {list(got.keys()) if isinstance(got, dict) else type(got)}"}
