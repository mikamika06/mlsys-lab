import ref

def check(workdir):
    from kvquant.calibrate import absmax_calibrate
    acts = ref.generate_activations()
    want = ref.absmax_calibrate(acts)
    out = {"absmax_matched": 0.0}
    try:
        got = absmax_calibrate(acts)
        if abs(got - want) < 1e-5:
            out["absmax_matched"] = 1.0
        else:
            out["_note"] = f"got calibration {got}, want {want}"
    except Exception as e:
        out["_note"] = f"exception during calibration: {type(e).__name__}: {str(e)[:100]}"
    return out
