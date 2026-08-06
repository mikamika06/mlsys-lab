import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from trtplan import parse_plan_header
    except ImportError as e:
        return {"headers_matched": 0.0, "_note": f"Import error: {e}"}
    finally:
        if workdir in sys.path:
            sys.path.remove(workdir)

    matched = 0
    for vec in ref.HEADER_VECTORS:
        want = ref.ref_parse_plan_header(vec)
        try:
            got = parse_plan_header(vec)
            if got == want:
                matched += 1
        except Exception:
            pass

    return {"headers_matched": float(matched)}
