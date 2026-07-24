## Context

Quantization reduces the precision of neural‑network weights from floating point to a smaller numeric format such as int8 or fp16.  
The scaling factor that maps the quantised integer back to real values can be shared across different parts of the weight tensor. Three common sharing strategies are

* **per‑tensor** – one scalar for the entire tensor,
* **per‑channel** – one scale per output channel (the first dimension of a convolution or linear layer),
* **per‑group** – several consecutive channels share a single scale; the number of channels in a group is called the *group size*.

The shape of the scale tensor therefore encodes the granularity.  
Given a weight tensor shape $W \in \mathbb{R}^{C_{\text{out}}\times C_{\text{in}}\times k_h\times k_w}$ and a scale tensor shape $S$, we want to infer whether $S$ represents per‑tensor, per‑channel or per‑group quantisation and, in the last case, what the group size is.

## Task

Implement `classify_quant_granularity(weight_shape: Tuple[int, ...], scale_shape: Tuple[int, ...]) -> Tuple[str, Optional[int]]`:

```python
def classify_quant_granularity(weight_shape, scale_shape):
    ...
```

The function receives the shapes of a weight tensor and its corresponding scale tensor as tuples of integers.  
It must return a tuple `(granularity, group_size)` where

* `granularity` is one of `"per_tensor"`, `"per_channel"` or `"per_group"`;
* `group_size` is an integer when `granularity == "per_group"` and `None` otherwise.

The implementation should handle the following cases:

1. **Per‑tensor** – `scale_shape` is empty `()` or a single element `(1,)`.
2. **Per‑channel** – `scale_shape` has one element equal to the first dimension of `weight_shape`.
3. **Per‑group** – `scale_shape` has one element that divides the first dimension of `weight_shape`; the group size is `weight_shape[0] // scale_shape[0]`.

If none of these patterns match, raise a `ValueError`.

## Example

```python
# per‑tensor
classify_quant_granularity((64, 3, 7, 7), (1,))
# → ("per_tensor", None)

# per‑channel
classify_quant_granularity((128, 64, 3, 3), (128,))
# → ("per_channel", None)

# per‑group with group size 8
classify_quant_granularity((256, 128, 3, 3), (32,))
# → ("per_group", 8)
```

## What the gate checks

The grader evaluates a handful of representative shapes and verifies that the returned tuple matches the reference implementation exactly. No other metrics are considered.
