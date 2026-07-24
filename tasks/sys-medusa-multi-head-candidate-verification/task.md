## Context

Medusa-style speculative decoding uses several lightweight prediction heads to propose
future token sequences. A verifier compares each proposed sequence against the target
model distribution and accepts a prefix that is sufficiently likely.

For a candidate path $c = (t_1, t_2, \dots, t_k)$ and target probabilities
$p_i(t)$ at each position, a token is accepted when

$$
p_i(t_i) \geq \tau ,
$$

where $\tau$ is the verification threshold.

The accepted prefix length of a candidate is the largest $m$ such that every token in
the prefix satisfies the probability constraint:

$$
m = \max \{j : \forall i \leq j,\ p_i(t_i) \geq \tau\}.
$$

A verifier can compare multiple Medusa heads and select the candidate with the longest
accepted prefix. If multiple candidates have the same accepted length, the lower
candidate index wins to keep the result deterministic.

## Task

Implement `verify_medusa_candidates(candidates, target_probs, threshold)`.

The function receives:

- `candidates`: a list of candidate token paths. Each path is a list of integer token
  IDs.
- `target_probs`: a list of dictionaries. Entry $i$ maps token IDs to the target model
  probability for position $i$.
- `threshold`: a floating point acceptance threshold.

Return a tuple:

```python
(best_index, best_path, accepted_indices)
```

where:

- `best_index` is the index of the candidate with the longest accepted prefix.
- `best_path` is the accepted prefix token list from that candidate.
- `accepted_indices` is a list containing every candidate index whose accepted prefix
  length is greater than zero.

Candidates may be shorter than the available probability list. Missing token
probabilities count as rejection.

The implementation should evaluate each candidate independently and return ordinary
Python values only.

## Example

```python
candidates = [[4, 8, 9], [4, 7], [3, 2]]
target_probs = [
    {4: 0.9, 3: 0.8},
    {8: 0.7, 7: 0.2, 2: 0.6},
    {9: 0.4}
]

verify_medusa_candidates(candidates, target_probs, 0.5)

# (0, [4, 8], [0, 2])
```

## What the gate checks

The gate computes the expected result using an independent verifier algorithm and
compares the returned tuple exactly. The oracle checks accepted prefixes, candidate
selection, and accepted candidate indices across cases containing rejected tokens,
missing probabilities, and ties.

Only the exact verification behavior passes the gate.
