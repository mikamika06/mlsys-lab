## Context

When a large model does not fit in GPU memory, some layers are *offloaded* to
either CPU RAM or disk. Every time a non-GPU layer is needed during a forward
pass, its weights must be read from wherever they are stored. The total byte
traffic per forward pass is therefore the sum of the sizes of all layers
placed off-GPU.

Let there be $L$ layers $\ell_1, \dots, \ell_L$ and a placement function
$p : \{\ell_i\} \to \{\text{'gpu'}, \text{'cpu'}, \text{'disk'}\}$. The forward
pass iterates through layers in order; each non-GPU layer is *streamed* in once
from its source. The total traffic is

$$
T(p) = \sum_{i : p(\ell_i) \neq \text{'gpu'}} \text{size}(\ell_i) .
$$

This simple model ignores caching and pipelining — exactly one read per
offloaded layer per forward pass.

## Task

Implement `offload_byte_traffic(placement, layer_sizes)`:

```python
def offload_byte_traffic(
    placement: dict[str, str],
    layer_sizes: dict[str, int]
) -> tuple[int, int]:
    ...
```

Arguments:

- `placement`: a dict mapping layer name (str) to one of `'gpu'`, `'cpu'`,
  `'disk'`.
- `layer_sizes`: a dict mapping layer name (str) to its weight size in bytes
  (int).

Return a 2-tuple `(cpu_traffic, disk_traffic)`:

- `cpu_traffic`: total bytes read from CPU RAM (sum of sizes of layers where
  `placement[layer] == 'cpu'`).
- `disk_traffic`: total bytes read from disk (sum of sizes of layers where
  `placement[layer] == 'disk'`).

Layers placed on `'gpu'` contribute zero to both totals. Every layer in
`placement` is guaranteed to have a corresponding key in `layer_sizes` with a
positive integer size.

## Example

```python
placement = {
    "embed": "gpu",
    "block0": "cpu",
    "block1": "disk",
    "block2": "gpu",
    "lm_head": "disk",
}
layer_sizes = {
    "embed": 4_000_000,
    "block0": 100_000_000,
    "block1": 100_000_000,
    "block2": 100_000_000,
    "lm_head": 2_000_000,
}
cpu_traffic, disk_traffic = offload_byte_traffic(placement, layer_sizes)
# cpu_traffic == 100_000_000   (block0)
# disk_traffic == 102_000_000  (block1 + lm_head)
```

## What the gate checks

A single `exact_match` gate: the returned tuple must exactly match the oracle
output computed from the same inputs. The check runs five randomly generated
layer configurations with varying placements and sizes. If every case matches,
the gate passes.
