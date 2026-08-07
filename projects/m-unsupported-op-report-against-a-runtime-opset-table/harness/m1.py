import ref


def check(workdir):
    from opset.report import check_support

    out = {"reports_matched": 0.0}
    tests = [
        (ref.NODES_1, 11, ref.TABLE_1),
        (ref.NODES_1, 9, {"Conv": 10, "Squeeze": 10, "Resize": 10}),
        (ref.NODES_1, 10, {"Conv": 9, "Squeeze": 10}),
    ]
    ok = 0
    for i, (nodes, opset, table) in enumerate(tests):
        want = ref.check_support(nodes, opset, table)
        got = check_support(nodes, opset, table)
        if want == got:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: want {want}, got {got}"
    out["reports_matched"] = float(ok)
    return out
