import ref
import sys


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"metrics_valid": 0.0}
    try:
        from optimum_export.export_utils import compare_export_metrics
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out

    try:
        ref_res = ref.compare_export_metrics("dummy")
        got_res = compare_export_metrics("dummy")
    except Exception as e:
        out["_note"] = f"execution error: {e}"
        return out

    if not isinstance(got_res, dict):
        out["_note"] = "result is not a dictionary"
        return out

    required = ["cli_time", "cli_size", "manual_time", "manual_size", "time_ratio", "size_ratio"]
    for k in required:
        if k not in got_res:
            out["_note"] = f"missing key {k}"
            return out

    if abs(got_res["time_ratio"] - ref_res["time_ratio"]) < 1e-5 and got_res["cli_size"] == ref_res["cli_size"]:
        out["metrics_valid"] = 1.0
    else:
        out["_note"] = f"metrics mismatch: got {got_res}, ref {ref_res}"
    return out
