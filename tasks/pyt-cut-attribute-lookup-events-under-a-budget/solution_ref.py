def accumulate_metric(model, steps):
    value = model.state.block.weight.value
    total = 0
    for _ in range(steps):
        total += value
    return total
