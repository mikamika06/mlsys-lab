import ref


def check(workdir):
    from torchprof.guards import check_recompiles
    model = ref.SimpleModel()
    inputs_list = ref.generate_dynamic_inputs()

    try:
        res = check_recompiles(model, inputs_list)
    except Exception as e:
        return {"guard_detected": 0.0, "_note": f"raised {type(e).__name__}: {e}"}

    if not isinstance(res, dict) or "recompile_count" not in res:
        return {"guard_detected": 0.0, "_note": "invalid return format"}

    if res["recompile_count"] >= 1:
        return {"guard_detected": 1.0}
    return {"guard_detected": 0.0, "_note": f"recompile count was {res['recompile_count']}"}
