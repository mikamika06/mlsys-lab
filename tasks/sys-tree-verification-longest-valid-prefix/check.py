def _oracle(tree, target):
    if not target:
        return []
    if tree.get("token") != target[0]:
        return []
    out = [tree["token"]]
    node = tree
    for expected in target[1:]:
        found = None
        for child in node.get("children", []):
            if child.get("token") == expected:
                found = child
                break
        if found is None:
            break
        out.append(found["token"])
        node = found
    return out


def _trees():
    return [
        (
            {
                "token": 1,
                "children": [
                    {"token": 2, "children": [{"token": 8, "children":[]}]},
                    {"token": 3, "children": []},
                ],
            },
            [1, 2, 8],
        ),
        (
            {
                "token": 5,
                "children": [
                    {"token": 7, "children": []},
                    {"token": 7, "children": [{"token": 9, "children":[]}]},
                ],
            },
            [5, 7, 9],
        ),
        (
            {
                "token": 4,
                "children": [
                    {
                        "token": 6,
                        "children": [
                            {"token": 10, "children":[]},
                            {"token": 11, "children":[]},
                        ],
                    }
                ],
            },
            [4, 6, 12],
        ),
        (
            {
                "token": 2,
                "children": [
                    {"token": 1, "children":[]},
                    {"token": 0, "children":[{"token": 3, "children":[]}]},
                ],
            },
            [2, 0, 3, 4],
        ),
        (
            {
                "token": 9,
                "children": [],
            },
            [8, 9],
        ),
    ]


def grade(sol, fx) -> dict:
    ok = 1.0
    for tree, target in _trees():
        try:
            got = sol.longest_valid_prefix(tree, list(target))
        except Exception:
            ok = 0.0
            break
        expected = _oracle(tree, target)
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
