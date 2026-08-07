import ref

def check(workdir):
    from modelfile.diff import semantic_diff
    out = {"diffs_matched": 0.0, "total": 3.0}
    ok = 0

    mf_a = "FROM llama3\n  PARAMETER temperature 0.7\n"
    mf_b = "  FROM llama3\nPARAMETER temperature 0.7  \n"
    d1 = semantic_diff(mf_a, mf_b)
    if not d1["added"] and not d1["removed"]:
        ok += 1

    mf_c = "FROM llama3\n"
    mf_d = "FROM llama3\nPARAMETER seed 42\n"
    d2 = semantic_diff(mf_c, mf_d)
    if len(d2["added"]) == 1 and not d2["removed"]:
        ok += 1

    d3 = semantic_diff(ref.MF1, ref.MF2)
    if len(d3["added"]) == 1 and len(d3["removed"]) == 1:
        ok += 1

    out["diffs_matched"] = float(ok)
    return out
