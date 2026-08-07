import ref


def check(workdir):
    from treeprune.heads import compare_heads

    out = {"budget_match": 0.0, "valid_comparison": 0.0}
    try:
        res = compare_heads(budget=1024)
    except Exception as e:
        out["_note"] = f"compare_heads raised {type(e).__name__}: {str(e)[:100]}"
        return out

    if not isinstance(res, dict):
        out["_note"] = "compare_heads must return a dictionary"
        return out

    med_p = res.get("medusa_params", 0)
    eagle_p = res.get("eagle_params", 0)

    if med_p == eagle_p and med_p > 0:
        out["budget_match"] = 1.0

    if "medusa_acc" in res and "eagle_acc" in res:
        out["valid_comparison"] = 1.0

    return out
