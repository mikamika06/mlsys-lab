import numpy as np
import ref


def check(workdir):
    from moe.block import MoEBlock

    out = {"forward_match": 0.0}
    matches = 0

    for i, cfg in enumerate(ref.CONFIGS):
        d_model = cfg["d_model"]
        d_ffn_fine = cfg["d_ffn_coarse"] // cfg["split_factor"]
        num_shared = cfg["num_shared"]
        num_routed = cfg["num_routed_fine"]
        top_k = cfg["k_fine"]

        ref_block = ref.ReferenceMoEBlock(d_model, d_ffn_fine, num_shared, num_routed, top_k, seed=100 + i)

        learner_block = MoEBlock(d_model, d_ffn_fine, num_shared, num_routed, top_k)
        learner_block.w_gate = ref_block.w_gate.copy()
        learner_block.shared_w1 = [w.copy() for w in ref_block.shared_w1]
        learner_block.shared_w2 = [w.copy() for w in ref_block.shared_w2]
        learner_block.shared_w3 = [w.copy() for w in ref_block.shared_w3]
        learner_block.routed_w1 = [w.copy() for w in ref_block.routed_w1]
        learner_block.routed_w2 = [w.copy() for w in ref_block.routed_w2]
        learner_block.routed_w3 = [w.copy() for w in ref_block.routed_w3]

        np.random.seed(200 + i)
        x = np.random.randn(4, d_model)

        want_out = ref_block.forward(x)
        got_out = learner_block.forward(x)

        if np.allclose(want_out, got_out, atol=1e-5):
            matches += 1

    out["forward_match"] = 1.0 if matches == len(ref.CONFIGS) else 0.0
    return out
