import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from sp_comm.comm_log import verify_comm_log
    except ImportError:
        sys.path.pop(0)
        return {"rel_err": 1.0, "_note": "failed to import verify_comm_log"}

    errs = []
    for cfg in ref.COMM_LOG_FIXTURES:
        try:
            want = ref.verify_comm_log(**cfg)
            got = verify_comm_log(**cfg)
            errs.append(abs(want - got))
        except Exception as e:
            sys.path.pop(0)
            return {"rel_err": 1.0, "_note": f"crashed on fixture: {e}"}

    sys.path.pop(0)
    return {"rel_err": sum(errs) / len(errs) if errs else 1.0}
