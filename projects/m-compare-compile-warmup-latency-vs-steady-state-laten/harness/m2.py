import ref

def check(workdir):
    from mpscompile.autotune import check_autotune_mode
    model = ref.SimpleModel()
    x = ref.get_test_inputs()[0]
    try:
        res = check_autotune_mode(model, x)
        valid = 1.0 if "status" in res else 0.0
    except Exception as e:
        return {"autotune_handled": 0.0, "_note": f"autotune check failed: {e}"}
    return {"autotune_handled": valid}
