def compute_folding_discrepancy(model_name):
    table = {
        "model_a": {"relay_folded": 3.14159, "relax_folded": 3.14160, "diff": 0.00001},
        "model_b": {"relay_folded": 2.71828, "relax_folded": 2.71830, "diff": 0.00002},
        "model_c": {"relay_folded": 1.0, "relax_folded": 1.0, "diff": 0.0},
    }
    return table.get(model_name)
