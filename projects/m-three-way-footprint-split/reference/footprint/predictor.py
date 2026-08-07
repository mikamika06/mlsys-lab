import math


def predict_peak_rss(execution_plan, alignment=64, overhead_bytes=0):
    """Predict peak RSS during model execution given tensor lifecycles."""
    if not execution_plan:
        return {
            "peak_rss_bytes": overhead_bytes,
            "peak_step": 0,
            "active_tensors_at_peak": []
        }

    max_step = 0
    for item in execution_plan:
        max_step = max(max_step, item.get("start_step", 0), item.get("end_step", 0))

    peak_rss = 0
    peak_step = 0
    peak_active = []

    for step in range(max_step + 1):
        current_rss = overhead_bytes
        current_active = []
        for item in execution_plan:
            start = item.get("start_step", 0)
            end = item.get("end_step", 0)
            if start <= step <= end:
                raw_size = item.get("size_bytes", 0)
                if alignment > 0:
                    aligned_size = math.ceil(raw_size / alignment) * alignment
                else:
                    aligned_size = raw_size
                current_rss += aligned_size
                current_active.append(item.get("name", "unnamed"))

        if current_rss > peak_rss:
            peak_rss = current_rss
            peak_step = step
            peak_active = sorted(current_active)

    return {
        "peak_rss_bytes": peak_rss,
        "peak_step": peak_step,
        "active_tensors_at_peak": peak_active
    }
