import numpy as np
import ref


def check(workdir):
    from moe_balance.bias_sim import simulate_deepseek_v3_bias_updates
    from moe_balance.compare import compare_convergence_speed

    out = {
        "bias_sim_matched": 0.0,
        "aux_vs_bias_convergence_matched": 0.0,
    }

    seq = ref.generate_synthetic_logits(num_batches=25, tokens=128, experts=8, seed=42)

    ref_sim = ref.simulate_deepseek_v3_bias_updates(seq, gamma=0.05, top_k=2)
    got_sim = simulate_deepseek_v3_bias_updates(seq, gamma=0.05, top_k=2)

    biases_close = np.allclose(ref_sim["biases"], got_sim["biases"], rtol=1e-5, atol=1e-5)
    loads_close = np.allclose(ref_sim["load_history"], got_sim["load_history"], rtol=1e-5, atol=1e-5)

    if biases_close and loads_close:
        out["bias_sim_matched"] = 1.0

    ref_cmp = ref.compare_convergence_speed(seq, alpha=0.01, gamma=0.05, top_k=1)
    got_cmp = compare_convergence_speed(seq, alpha=0.01, gamma=0.05, top_k=1)

    aux_cv_close = np.allclose(ref_cmp["aux_cv"], got_cmp["aux_cv"], rtol=1e-4, atol=1e-4)
    bias_cv_close = np.allclose(ref_cmp["bias_cv"], got_cmp["bias_cv"], rtol=1e-4, atol=1e-4)
    bool_match = ref_cmp["bias_converged_faster"] == got_cmp["bias_converged_faster"]

    if aux_cv_close and bias_cv_close and bool_match:
        out["aux_vs_bias_convergence_matched"] = 1.0

    return out
