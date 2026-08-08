import ref


def check(workdir):
    from diagnose.metrics import parse_metrics
    from diagnose.diff import compute_diff

    out = {"metrics_matched": 0.0}
    try:
        b_parsed = parse_metrics(ref.BASELINE_CSV)
        m_parsed = parse_metrics(ref.MASKED_CSV)
        diff = compute_diff(b_parsed, m_parsed)

        expected_keys = ["mcu_inst_executed", "stall_mio_throttle", "registers_per_thread", "sm_efficiency"]
        matched = 0
        for k in expected_keys:
            if k in diff and "ratio" in diff[k]:
                matched += 1
        out["metrics_matched"] = float(matched)
    except Exception as e:
        out["_note"] = f"Error during metrics parsing/diff: {str(e)[:120]}"
    return out
