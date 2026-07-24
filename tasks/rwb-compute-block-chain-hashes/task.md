## Context

In many distributed systems a *block chain* is built by repeatedly hashing the previous hash together with the contents of the current block.  
A common choice for the hash function in lightweight settings is a 64‑bit FNV‑1a style fold:

$$h_{i} = \bigl((\dots(((h_{i-1}\;\oplus\;t_0)\times P)\;\oplus\;t_1)\times P)\dots\bigr) \bmod 2^{64},$$

where $t_k$ are the token values of block $i$, $\oplus$ is bitwise XOR, and $P = 1099511628211$ is the FNV prime.  
The initial hash $h_0$ is a user‑supplied *salt*.

This task asks you to implement the per‑block hashing routine that follows this rule exactly.

## Task

Implement `compute_block_hashes`:

```python
def compute_block_hashes(tokens: np.ndarray, block_size: int, salt: np.uint64) -> np.ndarray:
    ...
```

- `tokens`: a 1‑D NumPy array of unsigned integers (any integer dtype).  
- `block_size`: positive integer; the last block may be shorter.  
- `salt`: initial hash value (`np.uint64`).  

The function must return a 1‑D `np.ndarray` of type `np.uint64`, where element $i$ is $h_i$ computed from the previous hash and the tokens in block $i$.  
All arithmetic should wrap modulo $2^{64}$ (NumPy’s unsigned integer types do this automatically).

## Example

```python
import numpy as np
tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
block_size = 2
salt = np.uint64(0x123456789ABCDEF0)

hashes = compute_block_hashes(tokens, block_size, salt)
print(hashes)
# array([<h_1>, <h_2>, <h_3>], dtype=uint64)
```

(The exact numeric values depend on the folding rule; see the grader for verification.)

## What the gate checks

The solution is graded by a deterministic oracle that recomputes the hash sequence using the same FNV‑1a folding rule.  
The metric `exact_match` compares your output array to the reference with element‑wise equality (`np.array_equal`).  
A perfect match yields a score of 1.0; any discrepancy gives 0.0.
