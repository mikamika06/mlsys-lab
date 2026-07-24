def _oracle_longest_match(tree, query):
    """Walk the tree token-by-token; return the number of matched tokens."""
    node = tree
    count = 0
    for token in query:
        if isinstance(node, dict) and token in node:
            node = node[token]
            count += 1
        else:
            break
    return count

def grade(sol, fx) -> dict:
    tree = {
        "a": {
            "b": {
                "c": {"d": {}},
                "e": {}
            },
            "f": {}
        },
        "g": {
            "h": {
                "i": {
                    "j": {"k": {}}
                }
            }
        },
        "m": {
            "n": {"o": {}},
            "p": {
                "q": {
                    "r": {
                        "s": {"t": {}}
                    }
                }
            }
        }
    }

    queries = [
        ["a", "b", "c", "d"],
        ["a", "b", "c"],
        ["a", "b", "x"],
        ["a"],
        ["z"],
        [],
        ["g", "h", "i", "j", "k"],
        ["g", "h", "i", "j", "x"],
        ["a", "f"],
        ["a", "b", "e"],
        ["m", "n", "o"],
        ["m", "n", "o", "p"],
        ["m", "p", "q", "r", "s", "t"],
        ["m", "p", "q", "r", "s", "t", "u"],
        ["x", "y", "z"],
        ["a", "b"],
        ["g"],
        ["g", "h"],
        ["m"],
        ["m", "p"],
    ]

    ok = 1.0
    for query in queries:
        expected = _oracle_longest_match(tree, query)
        try:
            got = sol.longest_match(tree, query)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
