import ref


def check(workdir):
    from marlin.validator import validate_quantize_config, QuantizationConfigError

    out = {"validation_passed": 0.0}
    try:
        validate_quantize_config({"bits": 4, "group_size": 128, "sym": True})
        bad_cfg = {"bits": 7}
        try:
            validate_quantize_config(bad_cfg)
            out["_note"] = "Validator failed to catch invalid bits=7"
            return out
        except Exception:
            pass
        out["validation_passed"] = 1.0
    except Exception as e:
        out["_note"] = f"Validator raised unexpectedly on valid config: {e}"
    return out
