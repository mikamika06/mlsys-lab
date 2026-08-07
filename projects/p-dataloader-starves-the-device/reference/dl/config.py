def predict_workers(target_rate, item_rate):
    import math
    if item_rate <= 0:
        return 1
    return max(1, math.ceil(target_rate / item_rate))
