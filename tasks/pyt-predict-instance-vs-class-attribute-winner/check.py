def _oracle(cases):
    results = []
    for case in cases:
        class_value = case["class_value"]
        has_instance_value = case["has_instance_value"]
        instance_value = case["instance_value"]

        Dynamic = type("Dynamic", (), {})
        Dynamic.target = class_value
        obj = Dynamic()

        if has_instance_value:
            obj.target = instance_value

        _ = obj.target
        if "target" in obj.__dict__:
            results.append(0)
        else:
            results.append(1)

    return results


def grade(sol, fx) -> dict:
    cases = [
        {
            "class_value": 3,
            "has_instance_value": True,
            "instance_value": 7,
        },
        {
            "class_value": "class",
            "has_instance_value": False,
            "instance_value": "unused",
        },
        {
            "class_value": [1, 2],
            "has_instance_value": True,
            "instance_value": [9],
        },
        {
            "class_value": None,
            "has_instance_value": False,
            "instance_value": 0,
        },
        {
            "class_value": 42,
            "has_instance_value": True,
            "instance_value": 42,
        },
    ]

    try:
        got = list(sol.predict_attribute_winner(cases))
    except Exception:
        return {"exact_match": 0.0}

    ref = _oracle(cases)
    return {"exact_match": 1.0 if got == ref else 0.0}
