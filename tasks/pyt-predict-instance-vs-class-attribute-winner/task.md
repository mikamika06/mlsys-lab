## Context

Python attribute lookup checks an object's instance dictionary before falling back to
the class dictionary for normal attributes. If an attribute name exists in the
instance storage, the instance value wins.

For an object $x$ and attribute name $a$, the simplified lookup rule is:

$$
\mathrm{winner}(x, a) =
\begin{cases}
\mathrm{instance}, & a \in x.\texttt{\_\_dict\_\_} \\
\mathrm{class}, & a \notin x.\texttt{\_\_dict\_\_}
\end{cases}
$$

This task focuses on the common case of plain attributes stored in either an
instance dictionary or a class dictionary. Descriptors and inheritance are not
part of this task.

## Task

Implement `predict_attribute_winner(cases)`:

```python
def predict_attribute_winner(cases):
    ...
```

Each item in `cases` is a dictionary with:

- `"class_value"`: the value stored on the generated class.
- `"has_instance_value"`: a boolean indicating whether the generated instance
  receives an attribute with the same name.
- `"instance_value"`: the value stored on the instance when
  `"has_instance_value"` is true.

For every case, return an integer list where:

- `0` means the attribute lookup winner is the instance dictionary.
- `1` means the attribute lookup winner is the class dictionary.

The function must classify the storage source, not compare the returned values.

## Example

```python
cases = [
    {
        "class_value": 10,
        "has_instance_value": True,
        "instance_value": 99,
    },
    {
        "class_value": 20,
        "has_instance_value": False,
        "instance_value": None,
    },
]

predict_attribute_winner(cases)
# [0, 1]
```

## What the gate checks

The gate creates real Python objects and uses Python's attribute lookup behavior
and object dictionaries as the reference oracle. The returned list must exactly
match the oracle classification for all generated cases.
