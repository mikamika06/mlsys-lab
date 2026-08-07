import ref


def check(workdir):
    from reconcile.device import validate_device_target

    out = {"device_checks_passed": 0.0}
    devices = ref.VALID_DEVICES

    try:
        assert validate_device_target("CPU", devices) is True
        assert validate_device_target("GPU:0", devices) is True
    except Exception as e:
        out["_note"] = f"valid device check failed: {type(e).__name__}: {str(e)}"
        return out

    invalid_inputs = ["INVALID_DEVICE", "MY_GPU:0"]
    for inv in invalid_inputs:
        try:
            validate_device_target(inv, devices)
            out["_note"] = f"failed to raise error for invalid device: '{inv}'"
            return out
        except RuntimeError as e:
            if "is not in list of supported devices" not in str(e):
                out["_note"] = f"wrong error message for '{inv}': {str(e)}"
                return out

    try:
        validate_device_target("", devices)
        out["_note"] = "failed to raise error for empty device string"
        return out
    except RuntimeError as e:
        if "Cannot open empty device name string" not in str(e):
            out["_note"] = f"wrong error message for empty device: {str(e)}"
            return out

    out["device_checks_passed"] = 1.0
    return out
