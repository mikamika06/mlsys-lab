## Context

CPython's evaluation loop executes one bytecode instruction at a time,
pushing and popping a per-frame **value stack**. Since 3.11, every scalar
arithmetic operator (`+`, `-`, `*`, `**`, `//`, ...) compiles down to the
same unified `BINARY_OP` instruction (its sub-operation is an argument, not
a separate opcode) — so counting *executed* `BINARY_OP` instructions is a
direct, hardware-independent measure of how much value-stack arithmetic a
piece of code actually performs at run time, as opposed to how it looks on
the page.

Evaluating a degree-$n$ polynomial $p(x) = \sum_{i=0}^{n} c_i x^i$ the
"obvious" way — computing each $x^i$ with `**` and each term with `*` —
executes 3 `BINARY_OP`s per term ($x^i$, $c_i \cdot x^i$, and the running
`+=`), i.e. $\Theta(n)$ operations but with a factor of 3, **and** it
recomputes powers of $x$ from scratch every term. **Horner's rule**
rewrites the same polynomial as a nested product,

$$
p(x) = c_0 + x\bigl(c_1 + x\bigl(c_2 + \cdots + x(c_{n-1} + x\,c_n)\cdots\bigr)\bigr),
$$

which needs exactly one multiply and one add per step down from $c_n$ to
$c_0$: $n$ multiplications and $n$ additions, $2n$ `BINARY_OP`s total for
an $(n{+}1)$-term polynomial — no `**` at all.

## Task

Implement `horner_eval`:

```python
def horner_eval(coeffs: list[float], x: float) -> float:
    ...
```

- `coeffs[i]` is the coefficient of $x^i$ (`coeffs[0]` is the constant
  term, low-degree first), matching $p(x) = \sum_i \text{coeffs}[i] \cdot x^i$.
- Return $p(x)$, evaluated using **Horner's rule** — work down from the
  highest-degree coefficient, doing one multiply-by-`x` and one add per
  step, and never raising `x` to a power.

## Example

```python
coeffs = [1.0, -2.0, 3.0]   # p(x) = 1 - 2x + 3x^2
horner_eval(coeffs, 2.0)
# == 3.0*2.0**2 - 2.0*2.0 + 1.0 == 9.0
# computed as ((3.0*2.0) - 2.0)*2.0 + 1.0, i.e. 2 multiplies + 2 adds
```

## What the gate checks

The grader runs `horner_eval` under a `sys.settrace` tracer with
`frame.f_trace_opcodes = True`, inspects the raw opcode byte at
`frame.f_lasti` via `frame.f_code.co_code`, and — using `dis.opname` for
the real, version-matched opcode names, exactly as the `dis` module itself
would report them — counts every executed `BINARY_OP` event across the
whole call (not source lines: actual executed instructions, so a loop that
runs $n$ times is counted $n$ times over).

Two gates apply, evaluated over several polynomials of different degrees:

- **`op_ratio`** — executed `BINARY_OP` count divided by $2n+2$ (Horner's
  exact $2n$ plus a small allowance for incidental bookkeeping arithmetic
  such as an index computation), must be `<= 1.0`. The naive
  power-then-multiply-then-add approach executes roughly $3(n+1)$
  `BINARY_OP`s and fails this comfortably at every tested degree.
- **`rel_err`** — the returned value, compared against `numpy.polyval` on
  the same coefficients (a real, independent oracle), must match to a
  relative error `<= 1e-9`.
