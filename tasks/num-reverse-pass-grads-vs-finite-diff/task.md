## Context

A **Wengert list** (tape) records a computation as a flat sequence of
elementary operations, each referring only to *earlier* entries — so the
tape is already in topological order. If $v_0, \dots, v_{n-1}$ are the
inputs and $v_n, \dots, v_{N-1}$ are the intermediate/output values produced
by the tape, reverse-mode autodiff computes $\partial v_{N-1} / \partial v_i$
for every $i$ by walking the tape **backward**, propagating an adjoint
$\bar v_k = \partial v_{N-1} / \partial v_k$ from each node to the nodes it
was built from:

$$
\bar v_{N-1} = 1, \qquad
\bar v_a \mathrel{+}= \bar v_k \cdot \frac{\partial v_k}{\partial v_a}
\quad \text{for every argument } a \text{ of node } k.
$$

The `+=` matters: if a value is consumed by more than one later node (or
twice by the same node), its adjoint must **accumulate** contributions from
every consumer, not just keep the last one.

The four supported elementary ops and their local derivatives are:

| op | value | local derivative(s) |
|---|---|---|
| `add(a, b)` | $v_a + v_b$ | $\partial/\partial v_a = 1,\; \partial/\partial v_b = 1$ |
| `mul(a, b)` | $v_a \cdot v_b$ | $\partial/\partial v_a = v_b,\; \partial/\partial v_b = v_a$ |
| `sin(a)` | $\sin(v_a)$ | $\partial/\partial v_a = \cos(v_a)$ |
| `exp(a)` | $\exp(v_a)$ | $\partial/\partial v_a = \exp(v_a)$ |

## Task

Implement `backward_pass`:

```python
def backward_pass(tape, values, n_inputs):
    """tape: list of (op, args) with op in {'add','mul','sin','exp'} and
    args a tuple of indices into `values`, each strictly less than the
    node's own index (topological order).
    values: the full forward trace — values[0:n_inputs] are the leaf
    inputs, values[n_inputs + k] is the value tape[k] produced. len(values)
    == n_inputs + len(tape). values[-1] is the scalar output.
    Returns a list/array of length n_inputs: d(values[-1])/d(values[i])."""
```

Walk `tape` from the last entry to the first, maintaining an adjoint for
every index in `values` (seeded with `1.0` at the output), and add each
node's local-derivative contribution into its arguments' adjoints. Finally
return the adjoints of indices `0 .. n_inputs-1`.

## Example

```python
tape = [("mul", (0, 1)), ("sin", (2,)), ("add", (2, 3))]
# v2 = x0*x1 ; v3 = sin(v2) ; v4 = v2 + v3   (output = v4, v2 reused twice)
inputs = [0.6, 1.1]
values = [0.6, 1.1, 0.66, 0.6131, 1.2731]     # forward trace, precomputed
grad = backward_pass(tape, values, n_inputs=2)
# grad ≈ [d(v4)/d(x0), d(v4)/d(x1)]
```

## What the gate checks

The gate runs your function on four hand-built tapes (2–4 ops each, every
tape containing at least one value that is read by more than one later op,
including one node that uses the *same* argument twice), with the forward
trace `values` already computed by the grader. It compares your returned
gradient against a **central finite-difference** estimate

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}, \qquad h = 10^{-5},
$$

obtained by re-running the tape's forward pass with each input perturbed —
an oracle that never touches your analytic derivative rules. **rel_err** is
the relative $L_2$ error between your gradient and this numerical estimate,
maximised over the four tapes, and must satisfy
$\mathrm{rel\_err} \le 10^{-5}$.
