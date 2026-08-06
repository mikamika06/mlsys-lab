import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from moe_dist.routing import measure_sparsity_pathology, simulate_loss_free_routing

    logits_list = ref.generate_test_logits(num_layers=6, batch_seq=1000, num_experts=32, seed=999)

    ref_res = ref.measure_sparsity_pathology(logits_list, [1, 2, 4])
    got_res = measure_sparsity_pathology(logits_list, [1, 2, 4])

    ok = 1
    for k in [1, 2, 4]:
        for metric in ["cv", "peak_ratio", "starved_experts"]:
            if abs(ref_res[k][metric] - got_res[k][metric]) > 1e-4:
                ok = 0
                break

    if got_res[1]["cv"] <= got_res[4]["cv"]:
        ok = 0

    return {"pathology_matched": float(ok)}
