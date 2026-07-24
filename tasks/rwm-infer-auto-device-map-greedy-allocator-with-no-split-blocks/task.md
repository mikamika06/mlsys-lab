## Context

Large model loading systems need to decide where modules should be placed before moving parameters to devices. A device map assigns each module name to a device while respecting memory limits.

Assume each module has a memory requirement $m_i$ and each device has capacity $C_j$. A greedy allocator processes modules in a fixed order and places each allocation block on the first device with enough remaining capacity.

Some modules cannot be split across devices. A no-split group is treated as one allocation block with total memory

$$
M_g = \sum_{i \in g} m_i .
$$

All modules inside that group receive the same device assignment.

## Task

Implement `infer_auto_device_map(modules, max_memory, no_split_modules)`:

```python
def infer_auto_device_map(modules, max_memory, no_split_modules):
    ...
```

Arguments:

- `modules` is a list of `(name, size)` pairs in allocation order. `name` is a string and `size` is an integer memory requirement.
- `max_memory` is a dictionary mapping device names to integer capacities. Devices are considered in dictionary insertion order.
- `no_split_modules` is a list of lists of module names. Every listed group must be allocated to one device.

Return a dictionary mapping every module name to its assigned device.

The allocator rules are:

1. Build allocation blocks from the no-split groups and individual modules.
2. Blocks are processed by the position of their first module in `modules`.
3. For each block, select the first device whose remaining capacity is at least the block size.
4. Assign every module in the block to that device.
5. Raise `ValueError` when a block cannot fit on any device.

## Example

```python
modules = [
    ("encoder.0", 40),
    ("encoder.1", 30),
    ("head", 20),
]
max_memory = {
    "cuda:0": 60,
    "cuda:1": 100,
}
no_split_modules = [
    ["encoder.0", "encoder.1"],
]

infer_auto_device_map(modules, max_memory, no_split_modules)
# {
#   "encoder.0": "cuda:1",
#   "encoder.1": "cuda:1",
#   "head": "cuda:0",
# }
```

## What the gate checks

The gate builds an independent greedy oracle and compares the returned mapping exactly.

The test cases include different module trees, budgets, and no-split groups. Implementations that split protected groups or use a different placement order will fail the exact-match gate.
