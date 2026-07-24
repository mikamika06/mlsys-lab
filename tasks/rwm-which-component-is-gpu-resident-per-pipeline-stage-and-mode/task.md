## Context

In a diffusion pipeline the computation is split into three stages:
encode, denoise and decode.  
A model may be **offloaded** between CPU and GPU in different ways.
Typical offload modes are:

- **model** – all components stay on the GPU throughout the whole run;
- **sequential** – only the component that is currently executed resides
  on the GPU, the others are moved to the CPU before the stage starts;
- **group** – a small group of components share GPU residency per stage,
  e.g. encode and denoise both keep the text encoder and denoiser on the
  GPU while decode keeps only the VAE.

The three components that can be resident are

```
text_encoder, denoiser, vae
```

## Task

Implement `residency(mode: str) -> dict[str, set[str]]`:

```python
def residency(mode: str) -> dict[str, set[str]]:
    ...
```

It receives a string `mode` that is one of `"model"`, `"sequential"` or
`"group"` and returns a mapping from each pipeline stage to the set of
components that are resident on the GPU for that mode.

The returned dictionary must contain exactly the keys

```
["encode", "denoise", "decode"]
```

and each value must be a `set[str]`.  The function should raise a
`ValueError` if an unknown mode is supplied.

## Example

```python
>>> residency("sequential")
{'encode': {'text_encoder'}, 'denoise': {'denoiser'}, 'decode': {'vae'}}
```

## What the gate checks

The grader computes the reference mapping from the same rules described
above and compares it with the candidate’s output using an exact set
equality test.  The metric `exact_match` must be `1.0`.  No other
performance or style constraints are enforced.
