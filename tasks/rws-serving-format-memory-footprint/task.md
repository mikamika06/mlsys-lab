## Context

NVIDIA's **2:4 structured sparsity** format keeps exactly 2 nonzero
values in every contiguous group of 4 (enabling 2x sparse tensor-core
throughput), but the *serving* footprint isn't just "half the values" —
you also have to store **metadata** saying *which* 2 of the 4 positions
survived, and if the kept values are themselves quantized (e.g. to
int4), you pay for **packing and a scale** too. This task computes the
actual served byte count for a given shape and config, and the resulting
compression ratio vs. a dense fp16 baseline.

### Format

For a `(d_out, d_in)` weight (`d_in` divisible by 4), split each row into
`d_in // 4` groups of 4 columns; each group keeps exactly 2 nonzero
values.

* **Kept values**: $\text{kept} = d_{out}\cdot(d_{in}/4)\cdot 2$.
* **Metadata**: each kept value needs a 2-bit code (its position, 0-3,
  within its group of 4), packed to whole bytes:
  $\text{meta\_bytes} = \lceil 2\cdot\text{kept} / 8\rceil$.
* **Values**:
  - if `use_int4`: kept values are int4-packed, 2 nibbles per byte
    ($\lceil \text{kept}/2\rceil$ bytes), plus **one fp16 scale per row**
    ($d_{out}\cdot 2$ bytes).
  - else: kept values are stored as fp16 ($\text{kept}\cdot 2$ bytes), no
    scale needed.

$$
\text{total} = \text{value\_bytes} + \text{meta\_bytes} + \text{scale\_bytes}
$$

$$
\text{size\_ratio} = \frac{d_{out}\cdot d_{in}\cdot 2}{\text{total}} \qquad (\text{dense fp16 bytes} / \text{total})
$$

## Task

Implement `sparse_2_4_footprint`:

```python
def sparse_2_4_footprint(d_out: int, d_in: int, use_int4: bool) -> tuple[int, float]:
    ...
```

* `d_out`, `d_in` — weight shape (`d_in % 4 == 0`).
* `use_int4` — whether the kept values are int4-packed (`True`) or kept
  as fp16 (`False`).

Return `(total_bytes, size_ratio)`, computed exactly as defined above.

## Example

```python
total, ratio = sparse_2_4_footprint(d_out=64, d_in=128, use_int4=True)
# kept = 64*32*2 = 4096; meta = ceil(8192/8) = 1024
# values = ceil(4096/2) = 2048; scales = 64*2 = 128
# total = 2048+1024+128 = 3200; ratio = (64*128*2)/3200 = 5.12
```

## What the gate checks

* **bytes_exact_match** — your `total_bytes` must exactly match a
  formula oracle, over a fixed set of `(d_out, d_in, use_int4)`
  configurations.
* **ratio_rel_err** — relative error between your `size_ratio` and the
  oracle's, on the same configurations.
