## Context

Sequence models often receive batches containing sequences with different lengths. A padded dense representation stores every sequence with the maximum length and uses a mask to ignore padding tokens. A ragged representation stores only valid tokens and keeps cumulative sequence lengths.

For attention, a sequence of length $L$ has an attention score matrix with $L^2$ token pairs. For a batch of sequence lengths $l_1, l_2, \dots, l_n$, ragged attention computes only valid pairs:

$$P_{\mathrm{ragged}} = \sum_{i=1}^{n} l_i^2 .$$

A padded dense batch uses the maximum length $L_{\max}$ for every sequence:

$$P_{\mathrm{padded}} = n L_{\max}^2 .$$

The pad waste ratio is

$$R = \frac{P_{\mathrm{padded}}}{P_{\mathrm{ragged}}}.$$

Production attention kernels use the ragged form internally through metadata such as cumulative sequence lengths (`cu_seqlens`) while returning the same outputs for valid tokens as a correctly masked dense implementation.

## Task

Implement `ragged_attention_compare(sequences)`:

```python
def ragged_attention_compare(sequences: list[list[list[float]]]) -> tuple[list[list[list[float]]], float]:
    ...
```

`sequences` is a list of list. Each array has shape $(L_i, d)$ and contains one sequence of query/key/value vectors. The function must return:

```python
valid_outputs, pad_waste_ratio
```

where:

- `valid_outputs` is a list of list, one per input sequence, containing the attention output for valid tokens only.
- `pad_waste_ratio` is the exact floating point value of $R$ computed from the sequence lengths.

Use the scaled dot-product attention formula:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.$$

For this task, each sequence array contains $Q=K=V$. Compute the output separately per sequence without padding. Do not include padded rows in the returned outputs.

## Example

```python

x = [[1.0, 0.0], [0.0, 1.0]]
y = [[1.0, 1.0]]

out, ratio = ragged_attention_compare([x, y])

# out contains two arrays with shapes (2, 2) and (1, 2)
# ratio is (2 * 2**2) / (2**2 + 1**2)
```

## What the gate checks

The gate builds a Python oracle that runs masked padded dense attention and ragged attention in `float64`.

The `max_abs_err` metric compares the student's valid-token outputs with the oracle outputs and must satisfy

$$\max |x_{\mathrm{student}} - x_{\mathrm{oracle}}| < 10^{-5}.$$

The `size_ratio` metric checks that the returned pad waste ratio exactly matches the oracle value computed from the sequence lengths:

$$R_{\mathrm{student}} = R_{\mathrm{oracle}}.$$
