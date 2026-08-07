## Context

RigL updates a sparse neural network topology while keeping the number of active
weights fixed. A binary mask $m$ determines which weights are currently alive.
The effective sparse weights are

$$
w_{\mathrm{sparse}} = m \odot w,
$$

where $\odot$ is elementwise multiplication.

A RigL topology update has two stages. First, remove a fraction of the smallest
magnitude active weights. For active index $i$, the removal score is

$$
s_i = |w_i|.
$$

The smallest scores among currently active connections are dropped. Second, grow
the same number of new connections from inactive positions using gradient
information. For inactive index $j$, the growth score is

$$
g_j = |G_j|,
$$

where $G$ is the dense gradient tensor. The inactive positions with the largest
growth scores become active.

The update fraction controls how many currently live connections are replaced:

$$
k = \lfloor r \cdot \lVert m \rVert_0 \rfloor ,
$$

where $r$ is the update fraction and $\lVert m \rVert_0$ is the number of active
mask entries. The total number of live weights remains unchanged.

## Task

Implement `rigl_topology_update(w, grad, mask, update_fraction)`.

```python
def rigl_topology_update(w, grad, mask, update_fraction):
    ...
```

The inputs are list of floats of equal length.

Return a new integer mask array with the same shape. The algorithm is:

1. Copy the input mask.
2. Compute $k = \lfloor r \cdot \lVert m \rVert_0 \rfloor$.
3. Among active entries, deactivate the $k$ entries with smallest $|w_i|$.
4. Among inactive entries after dropping, activate the $k$ entries with largest $|G_i|$.
5. Resolve all ties deterministically by preferring smaller indices.

The returned mask must contain only `0` and `1` values and preserve the original
number of active entries.

## Example

```python

w = [0.1, 5.0, 0.2, 0.0]
g = [1.0, 0.0, 3.0, 2.0]
m = [1, 1, 0, 0]

new_mask = rigl_topology_update(w, g, m, 0.5)
# Drops index 0 and grows index 2:
# [0, 1, 1, 0]
```

## What the gate checks

The gate computes a Python oracle implementation of the RigL update on several
inputs and compares the returned mask exactly. The check also verifies that the
number of live entries is conserved after the update. Incorrect tie handling,
wrong drop/grow ordering, or changing the sparsity level fail the gate.
