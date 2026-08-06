import ref


def check(workdir):
    from autotune_metrics.analyzer import find_argmin_config
    lines, expected_cfg = ref.generate_logs(seed=456)
    got_cfg = find_argmin_config(lines)

    matched = 1.0 if got_cfg == expected_cfg else 0.0
    out = {"argmin_matched": matched}
    if matched == 0.0:
        out["_note"] = f"expected {expected_cfg}, got {got_cfg}"
    return out
