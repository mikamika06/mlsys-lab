## Context

Speculative decoding methods can represent multiple future token choices as a tree. A verifier compares each speculative branch against a trusted greedy path and accepts the longest consecutive prefix that remains valid.

Represent a speculation tree as nested Python dictionaries. Each node has:

- `"token"`: the token value at this node.
- `"children"`: a list of child nodes.

The verifier has a greedy target sequence $T = [t_0, t_1, \dots, t_{m-1}]$. Starting from the root, a child is valid at depth $i$ if its token equals $t_i$. Verification follows the matching child recursively. When several children have the same token, choose the first matching child in the list.

The accepted result is the longest path of tokens that matches the target sequence. If the root token does not match $t_0$, the accepted path is empty.

## Task

Implement `longest_valid_prefix(tree, target)`:

```python
def longest_valid_prefix(tree: dict, target: list) -> list:
    ...
```

Return a list containing the accepted token path.

The function must:

1. Compare tokens in tree order.
2. Follow only children whose token matches the next target token.
3. Stop at the first missing match or when the target sequence ends.
4. Return the matching tokens, not the tree nodes.

## Example

```python
tree = {
    "token": 1,
    "children": [
        {
            "token": 2,
            "children": [
                {"token": 4, "children": []}
            ]
        },
        {
            "token": 9,
            "children": []
        }
    ]
}

target = [1, 2, 3]

# returns [1, 2]
```

## What the gate checks

The gate generates several speculation trees and target sequences. It computes the accepted path with an independent verifier implementation and compares the submitted function result against that oracle.

The `exact_match` score is $1.0$ only when every generated case returns exactly the same accepted token list as the reference verifier.
