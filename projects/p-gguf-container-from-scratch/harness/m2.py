import w


def check(workdir):
    out = {"keys_match": 0.0, "values_match": 0.0, "types_kept": 0.0}
    if w.gguf_or_none() is None:
        return w.needs_gguf(out)
    from gguf_lab import read_metadata

    tmp = w.tmpdir()
    p = w.fixture(tmp, rich=True)
    got = read_metadata(p)
    ref = w.oracle_meta(p)

    got_keys = set(got) - {"general.alignment"}
    ref_keys = set(ref) - {"general.alignment"}
    out["keys_match"] = 1.0 if got_keys == ref_keys else 0.0

    bad = []
    for k in sorted(ref_keys):
        a, b = got.get(k), ref[k]
        if isinstance(b, (list, tuple)):
            same = list(a or []) == list(b)
        elif isinstance(b, float):
            same = isinstance(a, float) and abs(a - b) < 1e-6
        else:
            same = a == b
        if not same:
            bad.append(f"{k}: {a!r} != {b!r}")
    out["values_match"] = 1.0 if not bad else 0.0
    out["types_kept"] = 1.0 if isinstance(got.get("lab.flag"), bool) and \
        isinstance(got.get("lab.name"), str) and isinstance(got.get("lab.list"), list) else 0.0
    if bad:
        out["_note"] = "; ".join(bad[:3])
    return out
