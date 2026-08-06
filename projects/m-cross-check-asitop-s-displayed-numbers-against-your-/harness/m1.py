import ref

def check(workdir):
    out = {"parsed_correct": 0.0}
    raw, disp, _, _ = ref.generate_fixtures()

    from edgemetrics.parser import cross_check, parse_powermetrics

    try:
        parsed = parse_powermetrics(raw)
        if parsed.get("gpu_power_mw") != disp["gpu_power_mw"]:
            out["_note"] = f"parse_powermetrics failed: got {parsed}, want {disp}"
            return out
        ok = cross_check(raw, disp)
        if ok:
            out["parsed_correct"] = 1.0
        else:
            out["_note"] = "cross_check returned False for matching data"
    except Exception as e:
        out["_note"] = f"Exception during execution: {type(e).__name__}: {str(e)}"
    return out
