def predict_attribute_winner(cases):
    result = []
    for case in cases:
        if case["has_instance_value"]:
            result.append(0)
        else:
            result.append(1)
    return result
