def evaluate_perplexity(model, remove_indices):
    base_ppl = 20.0
    ppl = base_ppl + len(remove_indices) * 2.5 + sum(remove_indices) * 0.1
    return float(ppl)
