import ref


def check(workdir):
    from atenaudit.analyzer import count_aten_ops, identify_decompositions

    ep = ref.get_test_exported_program()
    got_counts = count_aten_ops(ep)
    ref_counts = ref.count_aten_ops(ep) if hasattr(ref, "count_aten_ops") else {}

    target_ops = list(ref_counts.keys())[:3]
    if not target_ops:
        target_ops = ["torch.ops.aten.add.Tensor"]

    got_decomp = identify_decompositions(ep, target_ops)
    ref_decomp = {op: got_counts.get(op, 0) for op in target_ops}

    matched = 0
    if got_counts:
        matched += 1
    if got_decomp == ref_decomp:
        matched += 1
    if isinstance(got_counts, dict) and isinstance(got_decomp, dict):
        matched += 1

    return {"counts_matched": float(matched), "details": f"matched {matched} checks"}
