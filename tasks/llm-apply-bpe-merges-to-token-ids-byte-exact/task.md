## Context

Byte‑Pair Encoding (BPE) is a subword tokenisation technique that iteratively merges the most frequent adjacent symbol pair in a text corpus. In its simplest form each character can be regarded as an atomic symbol; after a sequence of merge operations the text is represented by longer symbols, which are then mapped to integer identifiers via a vocabulary.

The BPE algorithm proceeds left‑to‑right: whenever a specified pair $(s_1,s_2)$ occurs adjacent in the current token list it is replaced by the concatenated symbol $s_1s_2$. The process repeats until no merge rule applies. This deterministic procedure guarantees that two different implementations will produce exactly the same sequence of identifiers for a fixed text, merges and vocabulary.

## Task

Implement `apply_bpe_merges(text, merges, vocab)`:

```python
def apply_bpe_merges(text: str,
                     merges: list[tuple[str,str]],
                     vocab: dict[str,int]) -> list[int]:
    ...
```

* `text` – the raw string to be tokenised.
* `merges` – an ordered list of merge pairs. The order determines which pair is applied first; later pairs are considered only after earlier ones have been exhausted.
* `vocab` – a mapping from every symbol that may appear during the process (including single characters and any merged symbols) to its integer identifier.

The function must return a list of token identifiers corresponding to the fully‑merged representation of `text`. The implementation should use plain Python; NumPy is optional but allowed. No external libraries beyond the standard library are required.

## Example

```python
import numpy as np

vocab = {"a":1,"b":2,"c":3,"ab":4,"abc":5}
merges = [("a","b"), ("ab","c")]

ids = apply_bpe_merges("abc", merges, vocab)
print(ids)          # [5]
```

The string `"abc"` is first merged to `"ab"`, then the pair `("ab","c")` produces `"abc"`. The final identifier is `5`.

## What the gate checks

The grader compares the byte‑representation of the candidate’s output with a reference implementation. The metric used is **byte_exact_fraction** from `arena.scorers`; it returns `1.0` only when the two byte sequences are identical. Therefore any deviation in tokenisation order, missing merges or incorrect vocabulary look‑ups will cause the gate to fail.

The task has a single quality gate:

* **Metric:** `byte_exact_fraction`
* **Operator:** `>=`
* **Threshold:** `1.0`

Only an implementation that reproduces exactly the reference BPE tokenisation passes.
