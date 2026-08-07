def select_best_backend(rankings, fragmentation_cost, shape_churn_score):
    best_ep = rankings[0]
    if shape_churn_score > 0.7 and best_ep == "tensorrt":
        return "ort-cuda"
    if fragmentation_cost > 15.0 and best_ep == "tensorrt":
        return "ort-cuda"
    return best_ep
