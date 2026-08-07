import ref


def check(workdir):
    from kvparse.parser import parse_blocks
    out = {"parsed_correctly": 0.0}
    ok = 0
    for log_str, expected in ref.LOGS:
        got = parse_blocks(log_str)
        if got == expected:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"log '{log_str}': got {got}, expected {expected}"
    out["parsed_correctly"] = float(ok)
    return out
