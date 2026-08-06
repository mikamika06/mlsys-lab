def minimal_specializations(traces, budget):
    if not traces:
        return []
    unique = sorted(list(set(traces)))
    if len(unique) <= budget:
        return unique
    step = (len(unique) - 1) / float(budget - 1) if budget > 1 else 1
    specs = [unique[int(round(i * step))] for i in range(budget)]
    return sorted(list(set(specs)))
