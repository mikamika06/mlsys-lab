import ref


def check(workdir):
    from moe_quant.breakdown import compute_tensor_breakdown

    out = {"breakdown_matched": 0.0, "total_bytes_matched": 0.0}
    try:
        ref_res = compute_tensor_breakdown(ref.TENSORS)

        import moe_quant.breakdown as b
        user_res = b.compute_tensor_breakdown(ref.TENSORS)

        if user_res == ref_res:
            out["breakdown_matched"] = 1.0

        ref_total = sum(ref_res.values())
        user_total = sum(user_res.values()) if isinstance(user_res, dict) else -1
        if user_total == ref_total:
            out["total_bytes_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Milestone 1 execution error: {str(e)[:100]}"
    return out
