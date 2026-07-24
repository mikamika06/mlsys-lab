## Context

A regular expression describes a language: a set of strings that are accepted by
a finite-state machine. For a bounded search problem, we only need the portion
of the language whose strings satisfy a length constraint.

Let $L$ be the maximum length and let $\Sigma$ be the supplied vocabulary. The
goal is to compute

$$
S = \{ w \in \Sigma^{*} \mid |w| \leq L \text{ and } w \text{ is accepted by the regex} \}.
$$

A breadth-first exploration of the machine states can enumerate possible prefixes
while avoiding unbounded traversal. Each discovered accepting path corresponds
to one accepted string.

## Task

Implement `enumerate_accepted_strings(regex, vocab, max_len)`:

```python
def enumerate_accepted_strings(regex: str, vocab: list[str], max_len: int) -> list[str]:
    ...
```

The function receives a Python regular expression, a list of allowed symbols, and
a maximum string length. Return every string made from the symbols in `vocab`
whose length is at most `max_len` and which is accepted by the regex.

The return value must be a sorted list of unique strings. Matching must require
the entire string to be accepted, not only a substring.

The implementation should explore the finite search space incrementally rather
than generating strings of only the maximum length.

## Example

```python
out = enumerate_accepted_strings(r"ab+", ["a", "b"], 3)

# ["ab", "abb", "abbb"]
```

## What the gate checks

The gate runs the function on several regex, vocabulary, and length-bound
combinations. It computes the reference result independently by enumerating the
same finite language and using Python's regex engine for full-string matching.

The `exact_match` score is `1.0` only when the returned sorted list exactly
matches the oracle output for every case.
