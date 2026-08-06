import ref


def check(workdir):
    from atenaudit.capture import capture_global_mutation_error

    model, args = ref.get_mutation_func()
    res = capture_global_mutation_error(model, *args)

    ok = 0
    if isinstance(res, dict) and "error_type" in res and "message" in res:
        ok = 1
    elif res is not None and isinstance(res, dict):
        ok = 1

    return {"error_captured": float(ok)}
