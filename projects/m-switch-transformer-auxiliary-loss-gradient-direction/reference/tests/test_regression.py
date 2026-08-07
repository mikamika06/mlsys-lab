import numpy as np
from moe_balance.aux_loss import compute_switch_aux_loss
from moe_balance.bias_sim import simulate_deepseek_v3_bias_updates


def test_overallocated_expert_receives_negative_gradient():
    np.random.seed(42)
    logits = np.random.randn(100, 4)
    logits[:, 0] += 5.0  # Over-allocate expert 0
    _, grad = compute_switch_aux_loss(logits, alpha=0.01)

    mean_grad_exp0 = np.mean(grad[:, 0])
    mean_grad_exp1 = np.mean(grad[:, 1])

    # Overallocated expert must receive positive gradient wrt loss so subtraction decreases logit
    assert mean_grad_exp0 > mean_grad_exp1, "Overallocated expert should have higher positive gradient"


def test_bias_update_decreases_overloaded_expert_preference():
    np.random.seed(42)
    batch = np.zeros((100, 4))
    batch[:, 0] = 2.0  # Expert 0 heavily preferred
    seq = [batch.copy() for _ in range(10)]

    res = simulate_deepseek_v3_bias_updates(seq, gamma=0.1, top_k=1)
    biases = res["biases"]

    assert biases[-1, 0] < biases[0, 0], "Bias for overloaded expert must decrease"
    assert biases[-1, 1] > biases[0, 1], "Bias for underloaded expert must increase"
