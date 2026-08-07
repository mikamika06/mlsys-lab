"""Comparison between theoretical and real traces."""

def compare_traces(theoretical, real):
    diffs = [abs(t[1] - r[1]) for t, r in zip(theoretical, real)]
    mae = sum(diffs) / len(diffs) if diffs else 0.0
    max_diff = max(diffs) if diffs else 0
    return {"mae": mae, "max_diff": max_diff}
