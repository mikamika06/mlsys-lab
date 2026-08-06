import ref
import torch


def check(workdir):
    from partitioner.aot_bridge import compare_op_counts

    model = ref.get_test_module()
    x = torch.randn(4, 16)
    want = ref.compute_reference_op_counts()
    got = compare_op_counts(model, x)
    ok = 1.0 if got.get("standalone_ops") == want["standalone_ops"] and got.get("compiled_ops") <= want["compiled_ops"] else 0.0
    return {"op_count_matched": ok}
