import ref


def check(workdir):
    from chkpt import reproduce

    out = {"error_reproduced": 0.0}
    try:
        res = reproduce.trigger_error()
        if res is True or res == 1:
            out["error_reproduced"] = 1.0
        else:
            out["_note"] = f"trigger_error returned {res}, expected True"
    except Exception as e:
        out["_note"] = f"trigger_error raised {type(e).__name__}: {str(e)[:120]}"
    return out
