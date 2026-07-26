## Context

In IEEE‑754 floating point the *machine epsilon* $\varepsilon$ of a type $t$ is defined as the distance between $1$ and the next larger representable number in that type. For binary formats this spacing grows geometrically with the exponent, but at the value $1$ it equals the unit in the last place (ULP). The standard library exposes $\varepsilon$ via ``np.finfo(t).eps``, yet we can recover it directly by stepping from the bit pattern of ``1.0`` to its successor.

## Task

Implement `derive_eps()` that returns a tuple `(eps32, eps16, epsbf16)` containing the machine epsilon for 32‑bit float (`float32`), 16‑bit half precision (`float16`) and bfloat16 (`bfloat16`). The function must:

```python
def derive_eps() -> Tuple[float, float, float]:
    ...
```

The computation should be performed by *bit stepping*: obtain the next representable value after `1.0` in each dtype with ``np.nextafter`` (or an equivalent bit‑wise operation) and subtract `1`. The result must be a Python ``float`` or NumPy scalar of type ``float64``.

## Example

```python
>>> eps32, eps16, epsbf16 = derive_eps()
>>> eps32  # 2**-23 ≈ 1.1920929e‑07
1.1920928955078125e-07
>>> eps16  # 2**-10 ≈ 9.765625e‑04
0.0009765625
>>> epsbf16  # 2**-7 ≈ 7.8125e‑03
0.0078125
```

## What the gate checks

The grader compares your output to a reference computed with NumPy’s ``nextafter`` on the same machine. The relative L² error must satisfy

$$\mathrm{rel\_err} \le 10^{-12}.$$

A correct implementation will produce exactly the same bit patterns as the oracle, so the error is zero.
