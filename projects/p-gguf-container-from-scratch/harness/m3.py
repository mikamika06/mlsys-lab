import w


def check(workdir):
    out = {"names_match": 0.0, "shapes_match": 0.0, "offsets_aligned": 0.0, "offsets_increase": 0.0}
    if w.gguf_or_none() is None:
        return w.needs_gguf(out)
    from gguf_lab import read_tensor_info, read_metadata

    tmp = w.tmpdir()
    p = w.fixture(tmp, rich=True)
    got = read_tensor_info(p)
    ref = w.oracle_tensors(p)
    align = int(read_metadata(p).get("general.alignment", 32))

    out["names_match"] = 1.0 if [t["name"] for t in got] == [t["name"] for t in ref] else 0.0
    out["shapes_match"] = 1.0 if [list(t["shape"]) for t in got] == [list(t["shape"]) for t in ref] else 0.0
    offs = [t.get("offset") for t in got]
    out["offsets_aligned"] = 1.0 if all(isinstance(o, int) and o % align == 0 for o in offs) else 0.0
    out["offsets_increase"] = 1.0 if offs == sorted(offs) and len(set(offs)) == len(offs) else 0.0
    if not out["shapes_match"]:
        out["_note"] = f"shapes {[t.get('shape') for t in got]} vs {[list(t['shape']) for t in ref]}"
    return out
