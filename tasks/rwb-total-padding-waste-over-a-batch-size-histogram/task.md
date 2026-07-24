## Context

When processing data in mini‑batches, it is common to pad each batch so that all samples have the same length.  
If a batch contains $b$ samples and the padding unit (bucket) size is $p$, then the number of padded rows added to that batch is

$$\text{waste}(b)=\bigl(p-\,(b\bmod p)\bigr)\bmod p.$$

The first term computes how many rows are needed to reach a multiple of $p$; the outer modulus ensures that if $b$ is already a multiple of $p$, no padding is added.

Given a histogram that records how many batches have each size, the total padded rows wasted across all batches is

$$W=\sum_{b} \text{count}_b\;\text{waste}(b).$$

## Task

Implement `total_padding_waste`:

```python
def total_padding_waste(size_histogram: dict[int,int], bucket_size:int) -> int:
    ...
```

* `size_histogram` maps a batch size $b$ to the number of batches that have exactly $b$ samples.
* `bucket_size` is a positive integer $p$.
* Return the total number of padded rows wasted, as an `int`.

The function must be pure (no side effects) and work for any valid input.

## Example

```python
size_histogram = {3: 2, 5: 1}
bucket_size = 4
# waste(3)=1, waste(5)=3
# total = 2*1 + 1*3 = 5
print(total_padding_waste(size_histogram, bucket_size))  # → 5
```

## What the gate checks

The grader computes the reference value using the same formula and compares it to your output.  
Your implementation must return exactly the integer that the oracle produces; any deviation causes the `exact_match` gate to fail.
