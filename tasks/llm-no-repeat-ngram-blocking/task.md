## Context

In language model decoding we often want to avoid generating repeated phrases. A common technique is *no‑repeat n‑gram* blocking: when the decoder has already produced a sequence of tokens \(t_1,\dots,t_k\), any token that would complete an n‑gram that has appeared earlier in the same sequence is disallowed. Formally, for a fixed integer \(n \ge 1\) and a history \(h = (h_1,\dots,h_m)\), we ban every token \(x\) such that there exists an index \(i\) with \(1 \le i \le m-n+1\) for which

$$
(h_{i}, h_{i+1}, \dots, h_{i+n-2}) = (h_{m-n+2}, h_{m-n+3}, \dots, h_{m})
$$

and \(x = h_{i+n-1}\). In words: the current \((n-1)\)-token context (the last \(n-1\) tokens of the history, ending at \(h_m\)) already occurred earlier at position \(i\); the token \(h_{i+n-1}\) that followed it back then is banned, because appending it now would recreate that same n‑gram.

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
>>> prev = [3, 2, 4, 3, 2]
>>> no_repeat_ngram_blocking(prev, 3)
{4}
```

Explanation: The last two tokens (the current context, \(h_4, h_5\)) form the bigram \((3, 2)\). In the history there is one earlier occurrence of this bigram, at positions `1–2`, immediately followed by token `4`. Thus appending `4` would recreate the trigram \((3, 2, 4)\). No other token would produce such a repeat, so `4` is the only banned token.

## What the gate checks

The grader verifies that the returned set matches exactly the reference set computed by an oracle. A correct implementation must therefore return the exact set of banned tokens for any input history and \(n\).
