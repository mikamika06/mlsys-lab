import sys
import os


def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "reference"))
    import ref
    sys.path.pop(0)

    sys.path.insert(0, workdir)
    out = {"profiles_matched": 0.0, "size_ratio_valid": 0.0}
    try:
        from trtpipe.profile import analyze_build_tradeoffs
        
        want = ref.reference_analyze_build_tradeoffs(ref.SAMPLE_CONFIGS)
        got = analyze_build_tradeoffs(ref.SAMPLE_CONFIGS)

        if len(got) == len(want):
            matched = True
            ratio_ok = True
            for g, w in zip(got, want):
                if g.get("optimization_level") != w["optimization_level"]:
                    matched = False
                if abs(g.get("build_time_sec", 0) - w["build_time_sec"]) > 1e-5:
                    matched = False
                if g.get("plan_size_bytes") != w["plan_size_bytes"]:
                    matched = False
                if abs(g.get("size_ratio", 0) - w["size_ratio"]) > 1e-5:
                    ratio_ok = False
            
            if matched:
                out["profiles_matched"] = 1.0
            if ratio_ok:
                out["size_ratio_valid"] = 1.0
        else:
            out["_note"] = f"Expected {len(want)} profiled configs, got {len(got)}"
    except Exception as e:
        out["_note"] = f"Error during execution: {type(e).__name__}: {str(e)}"
    finally:
        if sys.path and sys.path[0] == workdir:
            sys.path.pop(0)
    return out
