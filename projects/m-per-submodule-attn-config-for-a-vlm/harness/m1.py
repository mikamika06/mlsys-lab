import ref


def check(workdir):
    from vlmcfg.parser import build_configs

    out = {"configs_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.build_configs(cfg)
        got = build_configs(cfg)
        norm = [{k: (sorted(v) if k == "submodules" else v) for k, v in g.items()
                 if k in ("kind", "num_heads", "kv_heads", "head_dim", "causal", "submodules")}
                for g in (got or [])]
        if norm == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {norm[:2]}, reference {want[:2]}"
    out["configs_matched"] = float(ok)
    return out
