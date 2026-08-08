import ref


def check(workdir):
    from surgery.extract import extract_subgraph

    base = ref.make_test_graph()
    test_cases = [
        ("input_x", "output_y"),
        ("t1", "t4"),
        ("t2", "output_y")
    ]
    ok = 0
    out = {"subgraphs_matched": 0.0}
    for inp, out_tensor in test_cases:
        want = ref.extract_subgraph(base, inp, out_tensor)
        got = extract_subgraph(base, inp, out_tensor)
        if got == want:
            ok += 1
        else:
            out["_note"] = f"mismatch for {inp} -> {out_tensor}"
    out["subgraphs_matched"] = float(ok)
    return out
