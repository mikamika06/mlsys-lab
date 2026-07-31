# ALiBi slopes and exact softcap for attention logits

A new model was switched from RoPE to ALiBi and got a softcap added on
attention logits (like in Gemma2 — so large logits don't push softmax into
zero-one). Right after the rollout: on contexts longer than the training
ones, distant tokens sometimes weigh MORE than nearby ones — even though
ALiBi is supposed to suppress exactly that. And on the backward pass, the
gradient through softcap sometimes lands outside `[0, 1]` and breaks the
gradient-norm check.

Both pieces were written in a hurry, without tests. We need to get the
slopes and the softcap right — and prove it with numbers, not by eyeballing
it.

## What you write

`scoremod/alibi.py` — `alibi_slopes(n_heads) -> np.ndarray` of shape
`(n_heads,)`. Each head gets its own slope for the positional penalty; when
`n_heads` is a power of two, it's a plain geometric sequence with a common
ratio; when it isn't, the base heads are computed for the nearest smaller
power of two, and the rest are picked from the sequence for double the
number of heads. All slopes are positive, no greater than 1, and different
heads get different values (and within a "pure" power-of-two group they're
also strictly decreasing).

`scoremod/softcap.py`:

```python
softcap_forward(x, cap) -> np.ndarray
softcap_backward(grad_output, x, cap) -> np.ndarray
```

`softcap_forward` caps the logit at `cap` (a scaled `tanh`), and for no
input can the result exceed `cap` in absolute value. `softcap_backward`
computes the gradient with respect to `x` via the chain rule; the local
derivative factor (the one `grad_output` gets multiplied by) always lies
in `(0, 1]`.

## How it's checked

The grader computes the reference itself — its own implementation of the
same formulas — over a set of head counts of varying parity (powers of two
and not), and over a set of caps/`x` values (some fixed, some drawn with a
fixed seed). The third milestone is yours: you write a test on the
`softcap.py` module, and we swap in a `softcap_backward` version that lost
the square in the `tanh` derivative. Your test needs to catch it.

```
mlsys project start m-alibi-slope-generator
mlsys project grade m-alibi-slope-generator --milestone 1
```
