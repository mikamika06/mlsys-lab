import ref


def check(workdir):
    from blobstore import build_blob_index

    out = {"index_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_blob_index(cfg)
        got = build_blob_index(cfg)
        norm = [{k: (sorted(v) if k == "tags" else v) for k, v in b.items()
                 if k in ("digest", "size", "tags")}
                for b in (got or [])]
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {norm[:2]}, reference {want[:2]}"
    out["index_matched"] = float(ok)
    return out
