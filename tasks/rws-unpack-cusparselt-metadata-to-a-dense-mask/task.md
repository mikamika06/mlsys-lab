## Context

cuSPARSELt stores sparse tensors in a compact format that packs the positions of non‑zero entries into a stream of bits.  
In this simplified setting each position is encoded with **two bits**:  

- `00` represents a zero entry,  
- any other value (`01`, `10`, or `11`) represents a one (non‑zero) entry.

The packed data is stored as an array of 8‑bit unsigned integers. Each byte contains four consecutive two‑bit codes, with the least significant pair describing the first position in that group.

Reconstructing the original dense mask therefore requires unpacking each byte into its four constituent two‑bit values and interpreting them as booleans.

## Task

Implement `unpack_cusparselt_metadata`:

```python
def unpack_cusparselt_metadata(metadata: np.ndarray, shape: tuple[int,int]) -> np.ndarray:
    ...
```

* `metadata` – a 1‑D NumPy array of dtype `uint8` containing the packed two‑bit codes.  
* `shape` – a tuple `(rows, cols)` specifying the desired output mask dimensions.

The function must return a dense boolean mask of shape `shape`. Bits that fall outside the requested size (due to padding in the last byte) should be discarded.

## Example

```python
import numpy as np
# Original mask
mask = np.array([[True, False, True],
                 [False, True, False]], dtype=bool)

# Packing (illustrative; not part of the task)
def pack(mask):
    flat = mask.ravel()
    pad_len = (-len(flat)) % 4
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=bool)])
    bytes_arr = []
    for i in range(0, len(flat), 4):
        v0,v1,v2,v3 = flat[i:i+4].astype(np.uint8)
        byte = int(v0 | (v1<<2) | (v2<<4) | (v3<<6))
        bytes_arr.append(byte)
    return np.array(bytes_arr, dtype=np.uint8)

metadata = pack(mask)

# Unpacking
D = unpack_cusparselt_metadata(metadata, mask.shape)
print(D)
```

Output:

```
[[ True False  True]
 [False  True False]]
```

## What the gate checks

The grader reconstructs a reference mask from random test cases using the same packing logic and compares it to the learner’s output.  
A single metric `exact_match` is used; the solution passes only if all reconstructed masks are **identical** to the originals (`threshold = 1.0`). The gate therefore guarantees that the unpacking algorithm is correct for any valid input.
