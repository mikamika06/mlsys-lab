## Context

Python attribute access follows a priority order. A data descriptor stored on the class is checked before an instance dictionary entry. If no data descriptor wins, an instance attribute can provide the value. A non-data descriptor is only used when the instance does not contain the same name. A normal class attribute is used after those checks, and `__getattr__` is the final fallback when normal lookup fails.

For an access name $x$, the lookup process selects exactly one winning storage location from the ordered set:

$$
\{\text{data descriptor},\ \text{instance dictionary},\ \text{non-data descriptor},\ \text{class attribute},\ \_\_getattr\_\_\}.
$$

This task asks you to classify the winner without performing the actual attribute access.

## Task

Implement `predict_storage_wins(accesses, class_dict, instance_dict, descriptor_flags)`.

Arguments:

- `accesses` is a list of attribute names to classify.
- `class_dict` is a list of names that exist in the class namespace.
- `instance_dict` is a dictionary of names that exist in the instance namespace.
- `descriptor_flags` maps class attribute names to one of `"data"`, `"nondata"`, or `"class"`.

Return a list of integers with one element per access:

- `0` for a data descriptor.
- `1` for an instance dictionary entry.
- `2` for a non-data descriptor.
- `3` for a normal class attribute.
- `4` for `__getattr__`.

Assume that every name marked as a descriptor is also present in `class_dict`.

## Example

```python
accesses = ["x", "y", "z", "missing"]

class_dict = ["x", "y", "z"]
instance_dict = {"x": 10, "y": 20}

descriptor_flags = {
    "x": "data",
    "y": "nondata",
    "z": "class",
}

predict_storage_wins(accesses, class_dict, instance_dict, descriptor_flags)
# [0, 1, 3, 4]
```

The instance value for `"x"` is ignored because a data descriptor has higher priority. The instance value for `"y"` wins because a non-data descriptor allows instance shadowing.

## What the gate checks

The gate creates real Python classes with data descriptors, non-data descriptors, class attributes, and `__getattr__`. It uses Python's attribute lookup behavior as the oracle and compares your returned labels against the oracle result.

The output must exactly match the oracle labels for every tested access sequence.
