def throughput_vs_budget(request_tokens, budgets):
    def tokens_processed(budget):
        running = 0
        for t in request_tokens:
            if running + t <= budget:
                running += t
        return running

    throughput_curve = [tokens_processed(b) for b in budgets]
    saturation_budget = sum(request_tokens)
    return throughput_curve, saturation_budget
