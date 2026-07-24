## Context

Instruction-level parallelism (ILP) depends on whether instructions can execute independently. Data hazards describe ordering constraints caused by accesses to the same storage locations.

For an instruction $i$ with read set $R_i$ and write set $W_i$, compare it with later instruction $j$ where $j > i$.

A read-after-write (RAW) hazard occurs when an earlier instruction writes a value that a later instruction reads:

$$W_i \cap R_j \ne \emptyset .$$

A write-after-read (WAR) hazard occurs when an earlier instruction reads a value that a later instruction overwrites:

$$R_i \cap W_j \ne \emptyset .$$

A write-after-write (WAW) hazard occurs when two instructions write the same value:

$$W_i \cap W_j \ne \emptyset .$$

Counting these dependencies helps identify which instruction pairs cannot be freely reordered.

## Task

Implement `detect_hazards(instructions)`:

```python
def detect_hazards(instructions):
    ...
```

The input is a list of dictionaries. Each dictionary represents one instruction and contains:

- `"reads"`: a list of integer register identifiers read by the instruction.
- `"writes"`: a list of integer register identifiers written by the instruction.

Return a tuple:

```python
(raw_count, war_count, waw_count)
```

Count every ordered instruction pair $(i, j)$ with $i < j$ that satisfies each hazard condition. If the same pair has multiple hazard types, count it in each applicable category.

The result must be deterministic and contain only integer counts.

## Example

```python
instructions = [
    {"reads": [], "writes": [1]},
    {"reads": [1], "writes": [2]},
    {"reads": [2], "writes": [1]},
]

detect_hazards(instructions)
# (2, 1, 0)
```

The first pair creates a RAW dependency on register $1$. The second pair creates a RAW dependency on register $2$. The first and third instructions create a WAR dependency because the first instruction writes register $1$ and the third instruction reads it while the second instruction reads the value before the overwrite.

## What the gate checks

The gate computes the hazard counts with a reference dependency analysis and compares the returned tuple exactly. The checker also uses a deterministic cache simulator oracle with fixed parameters to keep the evaluation environment independent of machine hardware behavior.

Only the returned hazard classification is graded.
