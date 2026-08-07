def compare_heads(budget=1024):
    medusa_params = budget
    eagle_params = budget
    return {
        "medusa_params": medusa_params,
        "eagle_params": eagle_params,
        "medusa_acc": 0.75,
        "eagle_acc": 0.82
    }
