def build_recipe(sensitivities, layer_sizes, target_avg_bits, candidate_bits=(2, 4, 8)):
    num_layers = len(layer_sizes)
    c_bits = sorted(list(candidate_bits))
    total_params = sum(layer_sizes)
    max_bit_budget = float(target_avg_bits) * total_params

    dp = {0: (0.0, [])}
    for i in range(num_layers):
        next_dp = {}
        s_dict = sensitivities[i]
        n_i = layer_sizes[i]
        for cost, (sens_sum, path) in dp.items():
            for b in c_bits:
                new_cost = cost + n_i * b
                if new_cost <= max_bit_budget + 1e-9:
                    new_sens = sens_sum + s_dict[b]
                    if new_cost not in next_dp or new_sens < next_dp[new_cost][0]:
                        next_dp[new_cost] = (new_sens, path + [b])
        dp = next_dp

    if not dp:
        return [min(c_bits)] * num_layers

    best_cost = min(dp.keys(), key=lambda c: dp[c][0])
    return dp[best_cost][1]
