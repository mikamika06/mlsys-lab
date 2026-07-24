## Context

ONNX Runtime's static quantizer does not use the same integer scheme for
every tensor — it picks the scheme based on the tensor's **role**:

* **Weights** are quantized to `QInt8` (signed 8-bit), **symmetric**:
  the zero-point is forced to $0$ and the code range is restricted to
  $[-127, 127]$ (not $-128$), so the representable range is perfectly
  symmetric around zero. This is a good fit because trained weights are
  usually already roughly zero-centered, and a hard-wired $zp=0$ lets a
  matmul kernel skip the zero-point correction terms entirely.
* **Activations** are quantized to `QUInt8` (unsigned 8-bit),
  **asymmetric**: the zero-point is calibrated from the observed
  min/max of the activation data, and codes range over the full
  $[0, 255]$. This is necessary because activations after a ReLU (or
  similar) are one-sided and their min/max is data-dependent, not
  known in advance — an unsigned, calibrated zero-point uses the whole
  8-bit range instead of wasting half of it on values that never occur.

This weight-symmetric / activation-asymmetric split is ONNX Runtime's
**default** `quantize_static` configuration and shows up throughout
production ONNX quantization pipelines.

## Task

Implement:

```python
def ort_default_scheme(tags: list[str]) -> list[tuple[int, int, bool]]:
    ...
```

* `tags` — a list of strings, each either `"weight"` or `"activation"`,
  one per tensor.

Return a list of the same length, one `(qmin, qmax, is_symmetric)` triple
per input tag:

* `tags[i] == "weight"` &rarr; `(-127, 127, True)`
* `tags[i] == "activation"` &rarr; `(0, 255, False)`

## Example

```python
ort_default_scheme(["weight", "activation", "weight"])
# [(-127, 127, True), (0, 255, False), (-127, 127, True)]
```

## What the gate checks

* **exact_match** — your list of `(qmin, qmax, is_symmetric)` triples
  must equal the oracle's, element for element, on several random tag
  sequences (fixed seed).
