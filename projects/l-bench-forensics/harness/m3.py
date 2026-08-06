import ref


def check(workdir):
    from benchkit import compare, parse

    rows = parse.load_all(ref.files())
    want = ref.raw()
    out = {"differences_match": 1.0, "controlled_found": 0.0, "confounded_found": 0.0}
    for i in range(0, len(rows), 3):
        for j in range(i + 1, len(rows), 5):
            if compare.differences(rows[i], rows[j]) != ref.expect_differences(
                    want[i], want[j]):
                out["differences_match"] = 0.0
    pairs = compare.controlled(rows, "n_ubatch")
    if pairs and all(ref.expect_differences(want[i], want[j]) == ["n_ubatch"]
                     for i, j in pairs):
        expected = sum(1 for i in range(len(want)) for j in range(i + 1, len(want))
                       if ref.expect_differences(want[i], want[j]) == ["n_ubatch"])
        if len(pairs) == expected:
            out["controlled_found"] = 1.0
    conf = compare.confounded(rows, "n_depth")
    if conf and all(len(extra) >= 1 for _, _, extra in conf):
        expected = sum(1 for i in range(len(want)) for j in range(i + 1, len(want))
                       if "n_depth" in ref.expect_differences(want[i], want[j])
                       and len(ref.expect_differences(want[i], want[j])) > 1)
        if len(conf) == expected:
            out["confounded_found"] = 1.0
    return out
