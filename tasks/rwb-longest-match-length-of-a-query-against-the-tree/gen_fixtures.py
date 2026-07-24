"""Generate fixture: a non-trivial trie and a battery of test queries."""
import json, os

tree = {
    "a": {
        "b": {
            "c": {
                "d": {}
            },
            "e": {}
        },
        "f": {}
    },
    "g": {
        "h": {
            "i": {
                "j": {
                    "k": {}
                }
            }
        }
    },
    "m": {
        "n": {
            "o": {}
        },
        "p": {
            "q": {
                "r": {
                    "s": {
                        "t": {}
                    }
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

os.makedirs(os.path.join(os.path.dirname(__file__), "fixtures"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "fixtures", "tree.json"), "w") as f:
    json.dump({"tree": tree, "queries": queries}, f, indent=2)
print("wrote fixtures/tree.json")
