import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from loraserve.config import compute_adapter_bytes
    import ref as reference_impl

    out = {"configs_matched": 0.0}
    matched = 0

    for item in reference_impl.CONFIG_TEMPLATES:
        peft_cfg = item["peft"]
        shapes = item["shapes"]

        expected = compute_adapter_bytes(peft_cfg, shapes)

        try:
            got = compute_adapter_bytes(peft_cfg, shapes)
            if got == expected:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Expected {expected}, got {got} for config {peft_cfg}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Error evaluating config: {type(e).__name__}: {str(e)}"

    out["configs_matched"] = float(matched)
    return out
