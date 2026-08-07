import ref


def check(workdir):
    from gparser.parser import parse_graph

    out = {"nodes_matched": 0.0}
    ok = 0
    for i, prog in enumerate(ref.PROGRAMS):
        want = ref.parse_graph(prog)
        got = parse_graph(prog)
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"program {i}: got {got}, reference {want}"
    out["nodes_matched"] = float(ok)
    return out
