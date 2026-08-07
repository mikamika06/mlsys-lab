import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    out = {"sidechannel_matched": 0.0, "audit_matched": 0.0}
    try:
        from vllm_sec.sidechannel import quantify_ttft_side_channel
        from vllm_sec.auditor import audit_launch_configs
    except ImportError as e:
        out["_note"] = f"ImportError: {e}"
        return out

    try:
        ref_sc = ref.quantify_ttft_side_channel(ref.TTFT_HITS, ref.TTFT_MISSES)
        got_sc = quantify_ttft_side_channel(ref.TTFT_HITS, ref.TTFT_MISSES)

        sc_ok = True
        for k in ref_sc:
            if abs(ref_sc[k] - got_sc.get(k, float("inf"))) > 1e-4:
                sc_ok = False
                out["_note"] = f"Sidechannel mismatch for key {k}: ref={ref_sc[k]}, got={got_sc.get(k)}"
                break
        if sc_ok:
            out["sidechannel_matched"] = 1.0

        ref_audit = ref.audit_launch_configs(ref.LAUNCH_CONFIGS)
        got_audit = audit_launch_configs(ref.LAUNCH_CONFIGS)

        if got_audit == ref_audit:
            out["audit_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"Audit mismatch: expected {ref_audit[:2]}, got {got_audit[:2]}"

    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {e}"

    return out
