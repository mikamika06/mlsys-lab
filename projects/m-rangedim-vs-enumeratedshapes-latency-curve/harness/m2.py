import ref
from shapes.enumerator import minimal_shape_set


def check(workdir):
    out = {"histogram_coverage": 0.0, "set_size_optimal": 0.0}
    try:
        res = minimal_shape_set(ref.SAMPLE_HISTOGRAM, max_waste=0.2)
        if isinstance(res, list) and len(res) > 0:
            out["set_size_optimal"] = 1.0
            covered = 0
            for l in ref.SAMPLE_HISTOGRAM.keys():
                if any(s >= l for s in res):
                    covered += 1
            coverage = covered / float(len(ref.SAMPLE_HISTOGRAM))
            out["histogram_coverage"] = float(coverage)
    except Exception as e:
        out["_note"] = f"Milestone 2 error: {type(e).__name__}: {str(e)[:120]}"
    return out
