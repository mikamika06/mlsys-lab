import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    ref.setup_mock_backends()
    out = {"probes_matched": 0.0}
    try:
        from flashsel.probe import probe_backend
        r_faulty = probe_backend("flashsel.backends.faulty")
        r_ideal = probe_backend("flashsel.backends.ideal")
        r_missing = probe_backend("flashsel.backends.nonexistent_xyz")
        if r_faulty is False and r_ideal is True and r_missing is False:
            out["probes_matched"] = 1.0
        else:
            out["_note"] = f"got faulty={r_faulty}, ideal={r_ideal}, missing={r_missing}"
    except Exception as e:
        out["_note"] = f"probe raised exception: {type(e).__name__}: {e}"
    return out
