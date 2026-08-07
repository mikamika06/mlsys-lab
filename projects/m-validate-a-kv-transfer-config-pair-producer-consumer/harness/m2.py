import sys
import os

def check(workdir):
    sys.path.insert(0, os.path.abspath(workdir))
    import ref

    try:
        from kvtransfer.triage import diagnose_stuck_handshake
        from kvtransfer.model import estimate_pipelined_transfer_time
    except Exception as e:
        return {
            "triage_accuracy": 0.0,
            "model_accuracy": 0.0,
            "_note": f"Import failed: {e}"
        }

    triage_ok = True
    for prod_logs, cons_logs in ref.LOG_TRIAGE_CASES:
        want = ref.diagnose_stuck_handshake(prod_logs, cons_logs)
        got = diagnose_stuck_handshake(prod_logs, cons_logs)
        if got.get("reason") != want["reason"] or got.get("stuck") != want["stuck"]:
            triage_ok = False
            break

    model_ok = True
    for model_cfg, net_cfg in ref.MODEL_CASES:
        want = ref.estimate_pipelined_transfer_time(model_cfg, net_cfg)
        got = estimate_pipelined_transfer_time(model_cfg, net_cfg)

        for key in ["sequential_ms", "pipelined_ms", "transfer_per_layer_ms", "speedup"]:
            if abs(got.get(key, 0.0) - want[key]) > 1e-4:
                model_ok = False
                break
        if not model_ok:
            break

    return {
        "triage_accuracy": 1.0 if triage_ok else 0.0,
        "model_accuracy": 1.0 if model_ok else 0.0
    }
