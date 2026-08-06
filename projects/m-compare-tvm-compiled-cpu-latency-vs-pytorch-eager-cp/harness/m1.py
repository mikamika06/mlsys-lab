import ref


def check(workdir):
    from tvm_bench.frontend import capture_frontend_error, import_and_compile

    out = {"supported_handled": 0.0, "unsupported_captured": 0.0}
    test_cases = ref.build_test_cases()

    supported_ok = True
    unsupported_ok = True

    for model, is_supported in test_cases:
        if is_supported:
            try:
                mod = import_and_compile(model, [1.0], ref.SUPPORTED_OPS)
                if mod is None:
                    supported_ok = False
            except Exception as e:
                supported_ok = False
                out["_note"] = f"Valid model failed import: {e}"
        else:
            captured, msg = capture_frontend_error(model, [1.0], ref.SUPPORTED_OPS)
            if not captured:
                unsupported_ok = False
                out["_note"] = f"Failed to capture unsupported op error: {msg}"

    if supported_ok:
        out["supported_handled"] = 1.0
    if unsupported_ok:
        out["unsupported_captured"] = 1.0

    return out
