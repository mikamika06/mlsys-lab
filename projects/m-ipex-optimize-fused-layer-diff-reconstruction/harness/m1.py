import ref


def check(workdir):
    from ipexdiff.diff import categorize_replacements, reconstruct_fused_diff

    out = {"models_diffed": 0.0}
    orig, opt = ref.generate_sample_models()

    diff = reconstruct_fused_diff(orig, opt)
    expected_keys = {"fc1", "conv1", "fc2"}
    if set(diff.keys()) != expected_keys:
        out["_note"] = f"Expected diff keys {expected_keys}, got {set(diff.keys())}"
        return out

    if diff["fc1"] != {"original": "Linear", "optimized": "IPEXLinearFused"}:
        out["_note"] = f"Unexpected diff structure for fc1: {diff['fc1']}"
        return out

    counts = categorize_replacements(diff)
    if counts.get(("Linear", "IPEXLinearFused")) != 2:
        out["_note"] = (
            f"Expected 2 Linear replacements, got {counts.get(('Linear', 'IPEXLinearFused'))}"
        )
        return out

    if counts.get(("Conv2d", "IPEXConv2dFused")) != 1:
        out["_note"] = (
            f"Expected 1 Conv2d replacement, got {counts.get(('Conv2d', 'IPEXConv2dFused'))}"
        )
        return out

    out["models_diffed"] = 1.0
    return out
