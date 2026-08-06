import ref


def check(workdir):
    from fsdp_balance.sharding import compute_load_balance
    out = {"balance_matched": 0.0, "total": float(len(ref.MODEL_SPECS) * 2)}
    ok = 0
    for spec in ref.MODEL_SPECS:
        for strategy in ["flat", "per_tensor"]:
            want = ref.compute_load_balance(spec["param_sizes"], spec["world_size"], strategy)
            try:
                got = compute_load_balance(spec["param_sizes"], spec["world_size"], strategy)
                if abs(got - want) < 1e-5:
                    ok += 1
            except Exception:
                pass
    out["balance_matched"] = float(ok)
    return out
