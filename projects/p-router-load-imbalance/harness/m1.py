import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from moe.metrics import compute_routing_entropy, measure_distribution
    except Exception:
        return {"distribution_measured": 0.0, "entropy_correct": 0.0}

    selected = np.array(
        [[0, 1], [0, 2], [0, 1], [3, 1], [0, 4], [0, 1], [5, 1], [0, 1]]
    )
    num_experts = 8

    learner_counts = measure_distribution(selected, num_experts)
    ref_counts = ref.get_ref_distribution(selected, num_experts)

    if not isinstance(learner_counts, np.ndarray) or not np.array_equal(
        learner_counts, ref_counts
    ):
        return {"distribution_measured": 0.0, "entropy_correct": 0.0}

    learner_entropy = compute_routing_entropy(learner_counts)
    ref_entropy = ref.get_ref_entropy(ref_counts)

    dist_ok = 1.0
    entropy_ok = 1.0 if abs(learner_entropy - ref_entropy) < 1e-4 else 0.0

    return {"distribution_measured": dist_ok, "entropy_correct": entropy_ok}
