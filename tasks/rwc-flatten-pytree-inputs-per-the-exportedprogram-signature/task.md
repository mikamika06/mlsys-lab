## Context

A packaged graph callable often does not receive the original user-facing input structure. Instead, it receives a flat positional argument list matching an exported graph signature.

An input pytree can contain nested dictionaries, lists, and tensors. The graph signature assigns each leaf a position and records how that leaf should be extracted. The flattening operation is therefore a reconstruction of the callable boundary:

$$
\mathrm{flat}(x) = [x_{p_0}, x_{p_1}, \dots, x_{p_{k-1}}],
$$

where each path $p_i$ identifies one leaf in the original structured input.

The ordering is controlled by the exported signature, not by dictionary traversal order or by the order of values discovered recursively. This allows parameters, buffers, and user inputs to appear in exactly the order expected by the generated graph.

## Task

Implement `flatten_exported_inputs(tree, input_spec)`.

```python
def flatten_exported_inputs(tree, input_spec):
    ...
```

`tree` is a nested Python structure containing dictionaries, lists, and list representing tensor leaves.

`input_spec` is a list of dictionaries. Each entry describes one flat argument:

```python
{
    "kind": "parameter" | "buffer" | "user_input",
    "path": ("key", 0, "name"),
}
```

The list order is the exact positional order expected by the exported callable.

Return a Python list containing the leaves selected by `path` for every entry in `input_spec`. Dictionary keys are strings in this task, and list path components are integer indices.

## Example

```python

tree = {
    "params": {"w": [1, 2]},
    "inputs": [[3, 4]]
}

spec = [
    {"kind": "parameter", "path": ("params", "w")},
    {"kind": "user_input", "path": ("inputs", 0)}
]

flat = flatten_exported_inputs(tree, spec)

# flat[0] is [1, 2]
# flat[1] is [3, 4]
```

## What the gate checks

The gate builds several exported-signature-like examples and computes the reference result by applying the signature paths to the structured input directly. The returned list must have the same length, order, and Python tensor contents as the oracle result.

A solution that recursively walks the pytree without respecting `input_spec` ordering will fail when the signature order differs from traversal order.
