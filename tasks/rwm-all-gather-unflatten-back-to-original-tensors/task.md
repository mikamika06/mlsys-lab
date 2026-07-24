## Context

Large model training systems often shard parameters across workers to reduce memory usage. During reconstruction, each worker contributes its shard through an all-gather operation. The gathered buffer contains the shards in worker order and may include padding because the shard size must be divisible across workers.

Suppose the original parameters are tensors $T_1, T_2, \dots, T_k$. They can be flattened into one vector:

$$
v = \mathrm{flatten}(T_1) \Vert \mathrm{flatten}(T_2) \Vert \dots \Vert \mathrm{flatten}(T_k),
$$

where $\Vert$ denotes concatenation. The flattened vector is divided into $N$ equal-sized shards. After all-gather, the shards are concatenated:

$$
g = s_0 \Vert s_1 \Vert \dots \Vert s_{N-1}.
$$

The gathered result can be larger than the original flattened vector because of padding. Reconstruction removes the padding, takes the first $|v|$ elements, and splits the vector back into the original tensor shapes.

## Task

Implement `unflatten_all_gathered`:

```python
def unflatten_all_gathered(shards: list[np.ndarray], shapes: list[tuple[int, ...]]) -> list[np.ndarray]:
    ...
```

The function receives:

- `shards`: the $N$ NumPy arrays produced by an all-gather operation. They have the same flattened dtype and represent consecutive pieces of the padded flattened parameter vector.
- `shapes`: the original parameter shapes in order.

Return a list of NumPy arrays with the requested shapes. The reconstruction algorithm is:

1. Concatenate all gathered shards.
2. Compute the number of elements required by `shapes`.
3. Remove any trailing padding elements.
4. Split the remaining flat vector according to the number of elements in each shape.
5. Reshape each segment into its original tensor shape.

The returned arrays must contain the original values and use NumPy reshaping rules.

## Example

```python
import numpy as np

shards = [
    np.array([1., 2., 3.]),
    np.array([4., 5., 0.]),
]

shapes = [(2,), (3,)]

params = unflatten_all_gathered(shards, shapes)

# params[0] == array([1., 2.])
# params[1] == array([3., 4., 5.])
```

The final zero is padding and is removed before splitting the parameters.

## What the gate checks

The gate builds original tensors with NumPy, flattens and shards them with padding, then uses the gathered shards as the input to the candidate implementation.

The reference reconstruction is computed by an oracle that concatenates the shards, strips the padded suffix, and splits and reshapes using the original shapes.

The reported metric is the maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_i |x_i - \hat{x}_i|.
$$

The implementation passes when $\mathrm{max\_abs\_err} \le 10^{-6}$ for all reconstructed parameters.
