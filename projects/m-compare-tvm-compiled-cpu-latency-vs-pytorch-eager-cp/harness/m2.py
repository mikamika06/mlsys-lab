import ref


def check(workdir):
    from tvm_compare.frontend import import_or_catch_error

    ok = 0
    total = len(ref.MODELS)
    for m in ref.MODELS:
        got = import_or_catch_error(m)
        unsupported = m.get("unsupported_op")
        if unsupported:
            if got.get("success") is False and got.get("op") == unsupported:
                ok += 1
        else:
            if got.get("success") is True:
                ok += 1

    out = {"error_captured": 1.0 if ok == total else 0.0}
    return out
