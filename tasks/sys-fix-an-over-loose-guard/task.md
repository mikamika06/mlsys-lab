## Context

Just-in-time tensor compilers such as PyTorch's `torch.compile` trace a
Python function once against a concrete example input and turn it into a
specialized compiled graph. Because that graph may bake in the input's
shape (loop bounds, buffer sizes, broadcast rules), the compiler cannot
safely reuse it for every future call. Instead, each compiled graph is
attached to a **guard**: a fast predicate over the new call's metadata
that must return `True` before the cached graph may be reused *as-is*. If
the guard returns `False`, the runtime must retrace ("recompile") against
the new input.

A guard that is *too loose* — one that says "reuse is safe" when it
actually isn't — is a real and dangerous class of bug: the runtime keeps
serving a stale compiled graph on inputs it was never specialized for,
silently producing wrong results (wrong output shape, or values computed
from a previous call's data).

This task uses a minimal single-slot cache. Each call is described by its
metadata

$$
\mu(x) = \big(\mathrm{shape}(x),\ \mathrm{dtype}(x)\big).
$$

A compiled graph traced on an input with metadata $\mu_{\text{cache}}$ is
safe to reuse for a new call with metadata $\mu_{\text{new}}$ **iff**

$$
\mathrm{shape}(x_{\text{cache}}) = \mathrm{shape}(x_{\text{new}})
\ \wedge\
\mathrm{dtype}(x_{\text{cache}}) = \mathrm{dtype}(x_{\text{new}}).
$$

The buggy guard below checks `dtype` correctly but, instead of comparing
the full shape tuple, only compares the **total element count**
($\mathrm{numel} = \prod_i \mathrm{shape}_i$). Two different shapes with
the same number of elements — e.g. $(2,6)$ and $(3,4)$, both $12$
elements — then incorrectly pass the guard, so a graph traced for $(2,6)$
gets reused, unchanged, for a call whose real input is shaped $(3,4)$.
Downstream code that reshapes the (stale) cached output back to
$\mathrm{shape}(x_{\text{cache}})$ then returns an array with the wrong
shape and, whenever the input values also changed, the wrong values.

## Task

Fix `guard_ok`:

```python
def guard_ok(cached_meta: dict, new_meta: dict) -> bool:
    ...
```

* `cached_meta`, `new_meta` — dicts with keys:
  * `"shape"` — a `tuple[int, ...]`.
  * `"dtype"` — a `str` (e.g. `"float64"`, `"int32"`).
* Return `True` only when the cached compiled graph may be safely reused
  for the new call, i.e. `shape` **and** `dtype` are both exactly equal.
  Return `False` (forcing a recompile) in every other case.

The current implementation is over-loose: it compares total element count
instead of the exact shape tuple, so it wrongly authorizes reuse whenever
two different shapes happen to have the same number of elements.

## Example

```python
cached = {"shape": (2, 6), "dtype": "float64"}

guard_ok(cached, {"shape": (2, 6), "dtype": "float64"})   # True  (identical)
guard_ok(cached, {"shape": (3, 4), "dtype": "float64"})   # False (shape differs, same numel)
guard_ok(cached, {"shape": (2, 6), "dtype": "int64"})     # False (dtype differs)
guard_ok(cached, {"shape": (2, 7), "dtype": "float64"})   # False (shape differs)
```

## What the gate checks

The grader drives a small single-slot compiled-graph cache through a
deterministic sequence of NumPy arrays with varying shapes and dtypes. For
each call it uses **your** `guard_ok` to decide whether to reuse the
cached (stale) output or recompute fresh output for the current input,
and records both the sequence of recompile decisions and the sequence of
returned arrays.

The same run is repeated with a reference guard that compares exact
`shape` and `dtype`. The single gate **exact_match** is `1.0` only if,
for every call, your recompile decision matches the reference decision
*and* your resulting output array matches the reference output array
exactly (same shape, same values) — otherwise it is `0.0`. Any exception
also counts as a failure.
