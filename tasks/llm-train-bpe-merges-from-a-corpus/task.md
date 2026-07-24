## Context

Byte-pair encoding (BPE) builds a vocabulary by repeatedly merging the most frequent
adjacent symbol pair in a corpus. A corpus is represented as a list of token
sequences, where each sequence is initially split into individual symbols.

For a sequence $s = (x_1, x_2, \dots, x_n)$, the frequency of a pair $(a,b)$ is

$$
f(a,b) = \sum_s \sum_{i=1}^{|s|-1} [s_i=a \land s_{i+1}=b],
$$

where $[P]$ is $1$ when proposition $P$ is true and $0$ otherwise.

One BPE training step selects the pair with the largest frequency and replaces
every occurrence of that adjacent pair with a new merged symbol. The ordered list
of selected pairs is the learned merge table.

When two pairs have the same frequency, this task uses deterministic
lexicographic ordering of the pair tuple to choose the smaller pair.

## Task

Implement `train_bpe_merges(corpus, num_merges)`:

```python
def train_bpe_merges(corpus: list[list[str]], num_merges: int) -> list[tuple[str, str]]:
    ...
```

The input `corpus` contains token sequences. Each sequence contains string symbols.
Start from the given sequences without modifying the caller's input.

For each of `num_merges` steps:

1. Count all adjacent symbol pairs across the current corpus.
2. Select the most frequent pair. Break ties by choosing the lexicographically
   smallest tuple.
3. Append that pair to the output merge list.
4. Replace every occurrence of the selected pair in every sequence with the
   concatenated symbol `left + right`.

Return the ordered list of selected merge pairs. If there are no adjacent pairs
left, stop early and return the merges found so far.

## Example

```python
corpus = [
    ["l", "o", "w"],
    ["l", "o", "w"],
    ["l", "o", "w", "er"],
]

merges = train_bpe_merges(corpus, 2)

# [
#   ("l", "o"),
#   ("lo", "w"),
# ]
```

## What the gate checks

The gate builds several fixed corpora and computes the expected merge sequence
using an independent greedy BPE implementation inside the grader. The returned
merge list must exactly match the oracle result.

The metric is `exact_match`. A score of $1.0$ means every merge pair and ordering
matches the reference algorithm.
