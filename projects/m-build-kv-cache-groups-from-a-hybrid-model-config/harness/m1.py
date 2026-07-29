import ref


def check(workdir):
    from kvplan import build_groups

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_groups(cfg)
        got = build_groups(cfg)
        norm = [{k: (sorted(v) if k == "layers" else v) for k, v in g.items()
                 if k in ("kind", "window", "kv_heads", "head_dim", "layers")}
                for g in (got or [])]
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {norm[:2]}, reference {want[:2]}"
    out["configs_matched"] = float(ok)
    return out
