def compute_attention_costs(lengths, max_length):
    padding_cost = len(lengths) * (max_length ** 2)
    packing_cost = sum(l ** 2 for l in lengths)
    return {"padding_cost": float(padding_cost), "packing_cost": float(packing_cost)}
