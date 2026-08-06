def perplexity_delta(data_free_ppl, awq_ppl):
    return float(data_free_ppl - awq_ppl)
