import ref


def check(workdir):
    out = {"validations_matched": 0.0}
    try:
        from tool_val.validator import validate_tool_call
    except Exception as e:
        out["_note"] = f"Import error: {type(e).__name__}: {e}"
        return out

    ok = 0
    total = len(ref.TEST_TOOL_CALLS)

    for i, tc in enumerate(ref.TEST_TOOL_CALLS):
        ref_valid, _ = ref.reference_validate_tool_call(tc, ref.TOOL_SCHEMAS)
        try:
            got_valid, got_errs = validate_tool_call(tc, ref.TOOL_SCHEMAS)
        except Exception as e:
            out["_note"] = f"Tool call {i} raised exception: {e}"
            return out

        if got_valid == ref_valid:
            if got_valid is False and not got_errs:
                out["_note"] = f"Tool call {i}: returned False but error list was empty"
                return out
            ok += 1
        else:
            out["_note"] = (
                f"Tool call {i} mismatch: expected valid={ref_valid}, got valid={got_valid}"
            )

    out["validations_matched"] = 1.0 if ok == total else 0.0
    return out
