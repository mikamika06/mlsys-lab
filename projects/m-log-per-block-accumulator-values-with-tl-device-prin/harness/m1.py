import ref

def check(workdir):
    import sys
    import os
    sys.path.insert(0, workdir)
    from debugger.logger import parse_device_print

    out = {"matches": 0.0}
    for log, expected in ref.generate_print_logs():
        try:
            got = parse_device_print(log)
            if got == expected:
                out["matches"] += 1.0
        except Exception:
            pass
    return out
