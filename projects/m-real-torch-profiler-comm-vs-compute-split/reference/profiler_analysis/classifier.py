from profiler_analysis.overlap import compute_overlap_percentage


def classify_overlap(trace):
    pct = compute_overlap_percentage(trace)
    return "enabled" if pct > 25.0 else "disabled"
