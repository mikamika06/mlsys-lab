## Context

A llama-style decoder processes multiple active sequences in one decode step. Each
active sequence contributes a token to a shared decode batch. The runtime converts
the per-sequence slot information into flat arrays consumed by the model.

For $N$ active slots, the packed token array is

$$
T = [t_0, t_1, \dots, t_{N-1}]
$$

where each position $i$ keeps the token metadata from the same input slot.

A llama batch stores several aligned arrays:

- `token`: token ids passed to the model.
- `pos`: token positions inside each sequence.
- `n_seq_id`: number of sequence ids attached to each token.
- `seq_id`: sequence id lists for each token.
- `logits`: flags indicating whether logits should be returned.

For this task every slot belongs to exactly one sequence, so each entry satisfies

$$
n\_seq\_id_i = 1
$$

and

$$
seq\_id_i = [s_i].
$$

The arrays must preserve the input slot ordering because the model output is aligned
with the packed token order.

## Task

Implement `pack_llama_batch(slots)`:

```python
def pack_llama_batch(slots):
    ...
```

`slots` is a list of dictionaries. Each dictionary has:

```python
{
    "token": int,
    "position": int,
    "seq_id": int,
    "wants_logits": bool
}
```

Return a dictionary with exactly these keys:

```python
{
    "token": [...],
    "pos": [...],
    "n_seq_id": [...],
    "seq_id": [...],
    "logits": [...]
}
```

Copy each slot into the corresponding output arrays in the same order. Every
`seq_id` output entry must be a one-element list.

## Example

```python
slots = [
    {"token": 42, "position": 5, "seq_id": 0, "wants_logits": True},
    {"token": 17, "position": 8, "seq_id": 3, "wants_logits": False},
]

pack_llama_batch(slots)
# {
#   "token": [42, 17],
#   "pos": [5, 8],
#   "n_seq_id": [1, 1],
#   "seq_id": [[0], [3]],
#   "logits": [True, False],
# }
```

## What the gate checks

The gate creates expected outputs with an independent reference algorithm and
compares all returned fields. The score `exact_match` must equal $1.0$.
