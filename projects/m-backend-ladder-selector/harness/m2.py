import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    ref.setup_mock_backends()
    out = {"ladder_matched": 0.0}
    try:
        from flashsel.selector import select_backend
        ladder = ["flashsel.backends.faulty", "flashsel.backends.ideal"]
        chosen = select_backend(ladder)
        if chosen == "flashsel.backends.ideal":
            out["ladder_matched"] = 1.0
        else:
            out["_note"] = f"expected ideal backend, got {chosen}"
    except Exception as e:
        out["_note"] = f"selector raised exception: {type(e).__name__}: {e}"
    return out
