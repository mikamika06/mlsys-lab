import numpy as np


def select_optimal_token_budget(budgets, workloads):
    best_budget = budgets[0]
    best_goodput = -1.0
    
    for b in budgets:
        itl_samples = []
        goodput_count = 0
        np.random.seed(42 + b)
        for w in workloads:
            prefill_len = w["prefill_len"]
            decode_len = w["decode_len"]
            effective_budget = max(64, min(b, 8192))
            prefill_chunks = max(1, int(np.ceil(prefill_len / effective_budget)))
            base_itl = 10.0 + (effective_budget / 512.0) * 3.0 + (prefill_chunks * 1.5)
            noise = np.random.normal(0, 2.0, decode_len)
            itls = base_itl + noise
            itl_p99 = float(np.percentile(itls, 99))
            itl_samples.append(itl_p99)
            if itl_p99 <= 40.0:
                goodput_count += decode_len
        
        overall_p99 = float(np.percentile(itl_samples, 99))
        if overall_p99 <= 40.0:
            if goodput_count > best_goodput:
                best_goodput = goodput_count
                best_budget = b
    return best_budget
