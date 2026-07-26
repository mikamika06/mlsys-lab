## Context

In a trie data structure each node represents a prefix of the stored strings. Two sequences share a non‑root prefix if they both start with the same token. Formally, for sequences
$$
a = (a_1,\dots,a_{|a|}), \qquad b = (b_1,\dots,b_{|b|}),
$$
they share a non‑root prefix iff $|a|>0$, $|b|>0$ and $a_1=b_1$. The root node corresponds to the empty prefix and is never considered a shared prefix.

## Task

Implement `shares_prefix(a, b)`:

```python
def shares_prefix(a: list[str], b: list[str]) -> bool:
    ...
```

It should return `True` when the two sequences share a non‑root prefix as defined above, otherwise `False`. The function must work for any hashable token type; strings are used in the examples.

## Example

```python
>>> shares_prefix(["apple", "banana"], ["apple", "cherry"])
True
>>> shares_prefix(["cat"], ["dog"])
False
>>> shares_prefix([], ["hello"])
False
```

## What the gate checks

The grader generates random pairs of token lists, computes the reference answer with the same rule, and compares your output. The metric `exact_match` must equal `1.0`; any mismatch or exception causes failure.
