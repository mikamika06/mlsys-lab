## Context

Python 3.11 introduced exception groups and the `except*` syntax for handling independent branches of failures. An `ExceptionGroup` contains multiple exceptions that can be split into matching and non-matching subgroups as each `except*` clause is evaluated.

This task uses a fixed handler chain:

1. `except* ValueError`
2. `except* LookupError`
3. `except* TypeError`
4. `except* Exception`

For a group of exceptions, the runtime evaluates these clauses in order. A branch fires if it receives at least one remaining exception after earlier branches have taken their matches. The output is an ordered list of branch indices.

The prediction problem is a classification function. If the input exception group is represented by labels $x_1, x_2, \dots, x_n$, the function predicts a vector

$$
y = [b_1, b_2, \dots, b_k]
$$

where each $b_i$ is the index of a branch that receives a non-empty subgroup during real `except*` execution.

## Task

Implement `predict_except_star(names)`:

```python
def predict_except_star(names: list[str]) -> list[int]:
    ...
```

The input is a list of exception class names. Supported names are:

- `"ValueError"`
- `"KeyError"`
- `"IndexError"`
- `"TypeError"`
- `"RuntimeError"`

Return the branch indices, in the exact order that the corresponding `except*` clauses execute.

The fixed branch mapping is:

```text
0 -> except* ValueError
1 -> except* LookupError
2 -> except* TypeError
3 -> except* Exception
```

Do not execute dynamic code or catch real exceptions to discover the answer. Implement the matching logic directly.

## Example

```python
predict_except_star(["ValueError", "KeyError", "RuntimeError"])
```

returns:

```python
[0, 1, 3]
```

The `ValueError` reaches branch 0, the `KeyError` reaches branch 1 through `LookupError`, and the remaining `RuntimeError` reaches branch 3.

## What the gate checks

The gate creates several `ExceptionGroup` objects and uses real CPython `except*` execution as the oracle. Your returned integer vector must exactly match the branch order observed from the interpreter.

The `exact_match` score is $1.0$ only when every tested group produces the same branch sequence as the real `except*` runtime.
