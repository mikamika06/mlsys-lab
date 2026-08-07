import ref

def check(workdir):
    from quantutil.serialize import serialize_quant_state
    from quantutil.reload import reload_quant_state

    out = {"states_matched": 0.0}
    ok = 0
    for i, state in enumerate(ref.CONFIGS):
        try:
            ser = serialize_quant_state(state)
            reloaded = reload_quant_state(ser)
            if getattr(reloaded, "bits", None) == state.bits and getattr(reloaded, "quant_type", None) == state.quant_type:
                ok += 1
        except Exception:
            pass
    out["states_matched"] = float(ok)
    return out
