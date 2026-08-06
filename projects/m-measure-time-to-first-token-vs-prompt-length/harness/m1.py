import ref


def check(workdir):
    from ttft.metrics import extract_ttft, aggregate_runs

    out = {"metrics_matched": 0.0}
    try:
        extracted = extract_ttft(ref.LOGS)
        aggregated = aggregate_runs(ref.RAW_DATA)
        ref_agg = aggregate_runs(extracted)

        if len(extracted) == len(ref.LOGS) and len(aggregated) == len(ref_agg):
            out["metrics_matched"] = 3.0
        else:
            out["_note"] = f"Extraction or aggregation size mismatch: got {len(extracted)} and {len(aggregated)}"
    except Exception as e:
        out["_note"] = f"Error in metrics execution: {type(e).__name__}: {str(e)[:120]}"
    return out
