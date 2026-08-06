import ref

def check(workdir):
    from isa.parser import analyze_objdump
    out = {"files_matched": 0.0, "files": 0.0}

    files = ref.generate_objdumps()
    out["files"] = float(len(files))
    ok = 0

    for name, lines in files.items():
        want = ref.analyze_objdump(lines)
        got = analyze_objdump(lines)
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"failed for {name}: want {want}, got {got}"

    out["files_matched"] = float(ok)
    return out
