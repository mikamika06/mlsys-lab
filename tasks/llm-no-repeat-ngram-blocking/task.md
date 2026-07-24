## Context

In language model decoding we often want to avoid generating repeated phrases. A common technique is *no‑repeat n‑gram* blocking: when the decoder has already produced a sequence of tokens \(t_1,\dots,t_k\), any token that would complete an n‑gram that has appeared earlier in the same sequence is disallowed. Formally, for a fixed integer \(n \ge 1\) and a history \(h = (h_1,\dots,h_m)\), we ban every token \(x\) such that there exists an index \(i < m-n+1\) with

$$
(h_{i}, h_{i+1}, \dots, h_{i+n-2}, x) = (h_{m-n+1}, h_{m-n+2}, \dots, h_{m-1}).
$$

This ensures that the next token cannot create a repeated n‑gram.

## Task

Implement `no_repeat_ngram_blocking(prev_tokens: List[int], n: int) -> Set[int]`:

```python
def no_repeat_ngram_blocking(prev_tokens: list[int], n: int) -> set[int]:
    ...
```

The function receives the sequence of token ids that have already been generated (`prev_tokens`) and an integer `n`. It must return a set containing all token ids that would create a repeated n‑gram if appended to `prev_tokens`. If no such tokens exist, the returned set should be empty.

The implementation must run in \(O(m)\) time where \(m = \lvert\text{prev\_tokens}\rvert\). No external libraries are required beyond the Python standard library.

## Example

```python
>>> prev = [5, 1, 3, 2, 4]
>>> no_repeat_ngram_blocking(prev, 3)
{4}
```

Explanation: The last two tokens form the bigram \((3, 2)\). In the history there is one occurrence of this bigram at positions `2–3` followed by token `4`. Thus appending `4` would create a repeated trigram. No other token would produce such an n‑gram.

## What the gate checks

The grader verifies that the returned set matches exactly the reference set computed by an oracle. A correct implementation must therefore return the exact set of banned tokens for any input history and \(n\).
