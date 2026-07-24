## Context

Python closures capture variables **by reference**, not by value. This is called
*late binding*: the variable name is resolved when the closure is **called**, not
when it is **defined**.

Consider a list comprehension that builds five lambdas:

```python
[lambda x: i * x for i in range(5)]
```

Each lambda captures the cell variable $i$ from the enclosing scope. Because all
five closures share the **same** cell, and the cell is updated on every
iteration, the final state after the loop is $i = 4$. At call time every closure
reads the same cell, so:

$$f_j(x) = 4 \cdot x \qquad \text{for all } j \in \{0,1,2,3,4\}$$

instead of the intended

$$f_j(x) = j \cdot x.$$

The classic fix forces **early binding** at definition time. Python evaluates
default arguments eagerly, so writing

$$\texttt{lambda } x,\; i{=}i \texttt{: } i \cdot x$$

creates a fresh function object per iteration with $i$ bound to the *current*
loop value. An alternative is an immediately-invoked factory:

$$(\lambda\, i.\; \lambda\, x.\; i \cdot x)(i)$$

Both strategies give each closure its own independent cell.

## Task

The file `starter.py` contains a broken `make_multipliers()` that exhibits the
late-binding closure bug. **Fix it** so the function returns a list of exactly
five callables $[f_0, f_1, f_2, f_3, f_4]$ satisfying:

$$f_i(x) = i \cdot x$$

for every integer $i \in \{0, 1, 2, 3, 4\}$ and any input $x$.

Do not change the function name or signature.

## Example

```python
multipliers = make_multipliers()
multipliers[0](3)  # → 0
multipliers[2](3)  # → 6
multipliers[4](3)  # → 12
```

A broken version returns `[4, 4, 4, 4, 4]` for every input because all five
lambdas share the loop variable's final value.

## What the gate checks

The grader calls each of the 5 returned functions with multiple integer inputs
$x$ and verifies

$$f_i(x) \stackrel{?}{=} i \cdot x$$

for all $i \in \{0, 1, 2, 3, 4\}$ and a range of test values including zero,
positives, and negatives. Every call must match exactly. A list that returns the
wrong count, contains non-callable elements, or produces incorrect results for
any pair $(i, x)$ fails the gate.
