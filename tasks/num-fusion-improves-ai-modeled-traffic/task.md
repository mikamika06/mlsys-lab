## Context

Arithmetic intensity describes how much computation is performed per byte moved from memory. The roofline model uses this relationship to explain why some workloads are limited by memory bandwidth rather than arithmetic throughput.

Consider an elementwise expression applied to $n$ values:

$$
y = (a+b)c+d .
$$

Without fusion, a system may materialize intermediate arrays:

$$
t=a+b,\qquad u=tc,\qquad y=u+d .
$$

Each intermediate write and later read increases memory traffic. If one element occupies $s$ bytes, the unfused chain moves:

$$
(3n + 3n + 3n)s = 9ns
$$

bytes, because each stage reads two inputs and writes one output.

With fusion, all operations can be evaluated in one pass:

$$
y_i=(a_i+b_i)c_i+d_i ,
$$

which moves only the original inputs and final output:

$$
(4n+n)s = 5ns .
$$

The access count is a simple model of memory traffic. It does not measure hardware cache behavior, but it allows comparing the effect of eliminating intermediate arrays.

## Task

Implement `model_access_count(n, element_size)`:

```python
def model_access_count(n: int, element_size: int) -> tuple[int, int]:
    ...
```

Return a tuple:

```python
(unfused_bytes, fused_bytes)
```

where:

- `unfused_bytes` is the modeled bytes moved for the three-stage unfused expression
  $y=(a+b)c+d$.
- `fused_bytes` is the modeled bytes moved when the chain is fused into one pass.

The model counts every array read and write exactly once. The number of elements is `n` and each element occupies `element_size` bytes.

## Example

```python
unfused, fused = model_access_count(1000, 8)

# unfused == 72000
# fused == 40000
```

## What the gate checks

The gate builds a reference model from the access rules in the context and compares the returned byte counts exactly on multiple inputs.

The metric `modeled_access_count` must be exactly `1.0` for the implementation to pass.
