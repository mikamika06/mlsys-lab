import numpy as np
import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from moe_dist.metrics import analyze_imbalance, compute_expert_load

    logits_list = ref.generate_test_logits(num_layers=2, batch_seq=500, num_experts=16, seed=123)
    max_err = 0.0

    for logits in logits_list:
        ref_counts = ref.compute_expert_load(logits, top_k=2)
        got_counts = compute_expert_load(logits, top_k=2)
        err_counts = np.max(np.abs(ref_counts - got_counts))
        max_err = max(max_err, float(err_counts))

        ref_metrics = ref.analyze_imbalance(ref_counts, 16)
        got_metrics = analyze_imbalance(got_counts, 16)

        for k in ["cv", "peak_ratio", "starved_experts"]:
            diff = abs(ref_metrics[k] - got_metrics[k])
            max_err = max(max_err, float(diff))

    return {"rel_err": float(max_err)}
