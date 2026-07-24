## Context

CPython dictionaries can use a split-table layout for instance attributes. When many instances of the same class have the same attribute names, the instances can share dictionary keys while storing only their values separately.

A combined dictionary stores both keys and values in each dictionary. A split instance dictionary stores a shared key table and per-instance values. The memory difference can be modeled as a ratio:

$$
\mathrm{savings\_ratio} = \frac{\mathrm{bytes\ of\ combined\ dicts}}{\mathrm{bytes\ of\ split\ instance\ dicts}} .
$$

A class using `__slots__` avoids instance dictionaries entirely by storing declared attributes in slot storage. This can be compared as a third layout:

$$
\mathrm{slot\ ratio} = \frac{\mathrm{bytes\ of\ combined\ dicts}}{\mathrm{bytes\ of\ slots}} .
$$

The exact byte sizes depend on the CPython runtime, so the implementation must measure the objects rather than use fixed constants.

## Task

Implement `model_split_dict_savings(m)`:

```python
def model_split_dict_savings(m: int) -> dict:
    ...
```

The function receives the number of instances to model. It must create real CPython objects and return a dictionary containing:

- `"split_bytes"`: total `sys.getsizeof(instance.__dict__)` bytes for instances of a class with shared instance attribute keys.
- `"combined_bytes"`: total `sys.getsizeof(d)` bytes for independent dictionaries containing the same attributes.
- `"slots_bytes"`: total `sys.getsizeof(instance)` bytes for instances of an equivalent `__slots__` class.
- `"savings_ratio"`: `combined_bytes / split_bytes`.

Use runtime measurement through `sys.getsizeof`. Do not return hardcoded measurements.

## Example

```python
result = model_split_dict_savings(100)

# result contains values similar to:
# {
#   "split_bytes": ...,
#   "combined_bytes": ...,
#   "slots_bytes": ...,
#   "savings_ratio": ...
# }
```

The exact numbers vary with the CPython build, but the ratio should be calculated from the measured sizes.

## What the gate checks

The gate creates the reference measurement using the CPython runtime itself. It compares the returned `savings_ratio` with an oracle that constructs split instance dictionaries, normal combined dictionaries, and slot instances using `sys.getsizeof`.

The result must match the oracle calculation for several instance counts. Returning fixed numbers or approximating dictionary sizes fails because the gate uses live CPython object measurements.
