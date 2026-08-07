import ref


def check(workdir):
    from kvmodel.dtypes import dtype_comparison_table

    out = {"table_match": 0.0, "savings_ratio": 0.0}
    cfg = ref.CONFIGS[0]
    want = ref.dtype_comparison_table(cfg, 16384, 1)
    got = dtype_comparison_table(cfg, 16384, 1)

    if got == want:
        out["table_match"] = 1.0
    else:
        out["_note"] = f"got table {got}, reference {want}"

    if got.get("fp16", 0) > got.get("int4", 0) * 3:
        out["savings_ratio"] = 1.0
    else:
        out["_note"] = "int4 savings ratio not achieved"

    return out
