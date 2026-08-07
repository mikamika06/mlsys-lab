import ref

def check(workdir):
    from precision.formats import get_format_props
    out = {"formats_matched": 0.0}
    ok = 0
    for name, expected in ref.FORMAT_TESTS:
        try:
            props = get_format_props(name)
            if isinstance(props, dict) and props.get("exp_bits") == expected["exp_bits"] and props.get("mantissa_bits") == expected["mantissa_bits"]:
                ok += 1
        except Exception:
            pass
    out["formats_matched"] = float(ok)
    return out
