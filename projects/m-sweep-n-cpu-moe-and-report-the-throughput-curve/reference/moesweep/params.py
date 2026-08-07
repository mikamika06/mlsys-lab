def activated_fraction(total_experts, active_experts, expert_hidden_dim, total_hidden_dim):
    expert_params = total_experts * expert_hidden_dim
    active_params = active_experts * expert_hidden_dim
    total = expert_params + total_hidden_dim
    active = active_params + total_hidden_dim
    return float(active) / float(total)
