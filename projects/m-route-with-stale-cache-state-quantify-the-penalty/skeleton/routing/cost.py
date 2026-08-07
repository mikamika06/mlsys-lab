def routing_cost(overlap_len, total_len, staleness, transfer_cost_per_token, compute_cost_per_token):
    raise NotImplementedError


def select_best_worker(request_tokens, workers, transfer_cost_per_token, compute_cost_per_token, current_tick):
    raise NotImplementedError
