import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from trtplan import parse_plan_header, classify_engine
    except ImportError as e:
        return {"status_matched": 0.0, "penalty_matched": 0.0, "_note": f"Import error: {e}"}
    finally:
        if workdir in sys.path:
            sys.path.remove(workdir)

    s_ok = 0
    p_ok = 0
    for raw_hdr, env in ref.CLASSIFY_VECTORS:
        hdr = ref.ref_parse_plan_header(raw_hdr)
        want = ref.ref_classify_engine(hdr, env)
        try:
            parsed = parse_plan_header(raw_hdr)
            got = classify_engine(parsed, env)
            if got.get("status") == want["status"]:
                s_ok += 1
            if got.get("penalty") == want["penalty"]:
                p_ok += 1
        except Exception:
            pass

    return {"status_matched": float(s_ok), "penalty_matched": float(p_ok)}
