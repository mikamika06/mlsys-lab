import numpy as np


def generate_shared_prefix_trace(num_requests, avg_prompt_len, avg_output_len, shared_ratio=0.60, seed=42):
    rng = np.random.default_rng(seed)
    prompt_lens = rng.poisson(lam=avg_prompt_len, size=num_requests)
    prompt_lens = np.maximum(prompt_lens, 10)

    total_prompt_tokens = int(np.sum(prompt_lens))
    target_shared_tokens = int(round(total_prompt_tokens * shared_ratio))

    prefix_pool_size = max(target_shared_tokens * 2, 500)
    prefix_pool = rng.integers(low=100, high=100000, size=prefix_pool_size).tolist()

    share_lengths = np.zeros(num_requests, dtype=int)
    current_shared = 0

    for i in range(num_requests):
        max_possible = min(prompt_lens[i] - 1, target_shared_tokens - current_shared)
        if max_possible > 0:
            if i == num_requests - 1:
                len_i = max_possible
            else:
                rem_reqs = num_requests - 1 - i
                len_i = int(round(target_shared_tokens / num_requests))
                len_i = min(len_i, max_possible)
            share_lengths[i] = len_i
            current_shared += len_i

    requests = []
    for i in range(num_requests):
        p_len = int(prompt_lens[i])
        s_len = int(share_lengths[i])
        unique_p_len = p_len - s_len

        prompt_tokens = prefix_pool[:s_len] + rng.integers(low=100, high=100000, size=unique_p_len).tolist()
        out_len = int(max(5, rng.poisson(lam=avg_output_len)))
        output_tokens = rng.integers(low=100, high=100000, size=out_len).tolist()

        requests.append({
            "request_id": f"req_{i}",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "shared_prefix_len": s_len
        })

    return requests
