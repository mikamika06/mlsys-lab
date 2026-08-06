import ref

def check(workdir):
    from occupancy.warp_stats import diagnose_gap
    out = {"diagnosis_matched": 0.0}
    match_count = 0
    for stats in ref.WARP_STATS_SAMPLES:
        got = diagnose_gap(0.85, 0.20, stats)
        want = ref.diagnose_gap(0.85, 0.20, stats)
        if got == want:
            match_count += 1
    if match_count == len(ref.WARP_STATS_SAMPLES):
        out["diagnosis_matched"] = 1.0
    else:
        out["_note"] = f"matched {match_count} of {len(ref.WARP_STATS_SAMPLES)} diagnoses"
    return out
