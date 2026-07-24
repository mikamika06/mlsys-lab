def _identity_oracle(objects):
    groups = {}
    for index, obj in enumerate(objects):
        key = id(obj)
        if key not in groups:
            groups[key] = []
        groups[key].append(index)
    ordered = sorted(groups.values(), key=lambda x: x[0])
    return tuple(tuple(group) for group in ordered)


def grade(sol, fx) -> dict:
    cases = []

    shared_list = [1, 2]
    cases.append([shared_list, [1, 2], shared_list])

    shared_dict = {"x": 1}
    cases.append([shared_dict, {"x": 1}, shared_dict, {"x": 1}])

    shared_tuple = (3, 4)
    cases.append([shared_tuple, shared_tuple, (3, 4)])

    x = object()
    y = object()
    cases.append([x, y, x, y, object()])

    ok = 1.0
    for objects in cases:
        try:
            got = sol.identity_partition(objects)
        except Exception:
            ok = 0.0
            break
        if got != _identity_oracle(objects):
            ok = 0.0
            break

    return {"exact_match": ok}
