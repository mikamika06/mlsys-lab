import ref


def check(workdir):
    from autoperf.nested import run_with_nested_disable

    out = {"nested_forces_fp32": 0.0}
    model, x = ref.get_fixture()
    try:
        states, _ = run_with_nested_disable(model, x)
        if isinstance(states, list) and len(states) >= 3:
            if states[0] is True and states[1] is False and states[2] is True:
                out["nested_forces_fp32"] = 1.0
            else:
                out["_note"] = f"Unexpected autocast state sequence: {states}"
        else:
            out["_note"] = "run_with_nested_disable did not return expected state list"
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {str(e)[:120]}"
    return out
