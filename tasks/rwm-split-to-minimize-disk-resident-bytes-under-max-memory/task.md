## Context

Large model runtimes often place parameter blocks across multiple memory tiers. A layer can be resident on an accelerator, resident in host memory, or left on disk. Each layer is treated as an indivisible block, so a placement must choose one location for the whole block.

For layer sizes $s_1, s_2, \dots, s_n$, GPU capacity $G$, and CPU capacity $C$, a placement chooses a location $x_i \in \{\mathrm{GPU}, \mathrm{CPU}, \mathrm{DISK}\}$ for each layer. The resident bytes are

$$
R = \sum_{i:x_i=\mathrm{GPU}} s_i + \sum_{i:x_i=\mathrm{CPU}} s_i .
$$

The goal is to minimize disk-resident bytes. This is equivalent to maximizing resident bytes while respecting the two memory limits:

$$
\sum_{i:x_i=\mathrm{GPU}} s_i \le G,\qquad
\sum_{i:x_i=\mathrm{CPU}} s_i \le C .
$$

A production placement pass solves this as a whole-block allocation problem. The optimal solution keeps as many bytes resident as possible and leaves only the unavoidable remainder on disk.

## Task

Implement `split_layers(layer_bytes, gpu_cap, cpu_cap)`:

```python
def split_layers(layer_bytes: list[int], gpu_cap: int, cpu_cap: int) -> list[int]:
    ...
```

Return a list with one integer per layer:

- `0` means place the layer on GPU.
- `1` means place the layer on CPU.
- `2` means leave the layer on disk.

Each layer must have exactly one placement. The returned placement must minimize the total bytes assigned to disk. Layers cannot be split between memory tiers.

If several placements have the same minimum disk usage, any one of them is accepted only if it has the same minimum disk-byte total as the oracle solution.

## Example

```python
layers = [8, 5, 4]
placement = split_layers(layers, 9, 5)

# One valid optimal answer:
# [0, 1, 2]
#
# GPU: 8 bytes
# CPU: 5 bytes
# Disk: 4 bytes
```

## What the gate checks

The gate computes an optimal whole-block allocation with a dynamic-programming oracle and compares the disk-byte total of the submitted placement against that oracle on generated layer arrays and memory capacities.

The returned list must contain only valid placement codes, and its disk usage must equal the oracle minimum.
