import ref

def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    from debugger.masking import extract_program_id

    out = {"matches": 0.0}
    for log, expected in ref.generate_error_logs():
        try:
            got = extract_program_id(log)
            if got == expected:
                out["matches"] += 1.0
        except Exception:
            pass
    return out
