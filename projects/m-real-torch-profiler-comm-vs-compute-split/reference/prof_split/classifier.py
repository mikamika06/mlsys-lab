from prof_split.analyzer import compute_overlap_percentage, extract_intervals


def classify_overlap(trace_data):
    comp, comm = extract_intervals(trace_data)
    pct = compute_overlap_percentage(comp, comm)
    return "overlap_enabled" if pct > 15.0 else "overlap_disabled"
