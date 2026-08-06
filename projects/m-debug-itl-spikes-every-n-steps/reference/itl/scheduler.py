def step_budget(step, n, budget):
    if step > 0 and step % n == 0:
        return budget // 2
    return budget
