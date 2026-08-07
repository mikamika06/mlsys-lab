"""Theoretical minimum step time and overlap efficiency metric calculation."""

from overlap.reconstruct import reconstruct_timeline


def compute_metrics(events):
    recon = reconstruct_timeline(events)
    total_compute = recon["compute_only"] + recon["overlapped"]
    total_comm = recon["comm_only"] + recon["overlapped"]

    theoretical_min_step_time = max(total_compute, total_comm)
    actual_step_time = recon["total_span"]

    if total_comm > 0:
        overlap_efficiency_score = recon["overlapped"] / total_comm
    else:
        overlap_efficiency_score = 1.0

    return {
        "theoretical_min_step_time": float(theoretical_min_step_time),
        "actual_step_time": float(actual_step_time),
        "overlap_efficiency_score": float(overlap_efficiency_score),
    }
