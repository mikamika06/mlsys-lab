import ref


def check(workdir):
    from radixkv.overhead import tree_memory_overhead
    out = {"overhead_rel_err": 0.0}
    errors = []
    for cfg in ref.CONFIGS:
        want = ref.tree_memory_overhead(cfg["node_count"], cfg["branch_factor"], cfg["metadata_bytes"])
        try:
            got = tree_memory_overhead(cfg["node_count"], cfg["branch_factor"], cfg["metadata_bytes"])
            err = abs(got - want) / max(1.0, float(want))
            errors.append(err)
        except Exception:
            errors.append(1.0)
    out["overhead_rel_err"] = float(max(errors)) if errors else 1.0
    return out
