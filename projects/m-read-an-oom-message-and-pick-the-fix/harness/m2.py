import ref


def check(workdir):
    from triage.allocator import tune_max_split_size

    out = {"optimal_split_matched": 0.0}
    for trace in ref.SAMPLE_TRACES:
        best = tune_max_split_size(trace, ref.CANDIDATE_SIZES)
        if best not in ref.CANDIDATE_SIZES:
            out["_note"] = f"tune_max_split_size returned {best}, not in candidates"
            return out

    out["optimal_split_matched"] = 1.0
    return out
