## Context

CPython frees many objects when their reference count reaches zero. A debugging task is to identify which reference edge keeps a target object alive.

A retaining edge is a concrete reference from a container or owner to the object. Removing that edge may reduce the object's reference count enough that a `weakref` becomes dead.

A graph edge can be viewed as a directed reference:

$$
\text{owner} \xrightarrow{\text{edge id}} \text{target object}.
$$

Different Python mechanisms create different edge types. A list slot keeps an object alive through an index, an attribute keeps it alive through a name, and a closure cell keeps it alive through a captured variable.

## Task

Implement `find_retaining_edge(graph)`.

The input is a dictionary with:

- `graph["target"]`: the object whose final retaining reference must be found.
- `graph["edges"]`: a list of edge descriptors.

Each edge descriptor has:

- `id`: the identifier to return.
- `kind`: one of `"list"`, `"attribute"`, or `"cell"`.
- `owner`: the object that contains the reference.
- For a list edge, `slot` is the integer list index.
- For an attribute edge, `name` is the attribute name.
- For a closure cell edge, `cell` is the cell object.

Return the `id` of the edge which, when cut, causes the target object's `weakref` to become dead. Do not permanently modify the graph.

Example:

```python
class Box:
    pass

obj = Box()
items = [obj]

graph = {
    "target": obj,
    "edges": [
        {
            "id": "items[0]",
            "kind": "list",
            "owner": items,
            "slot": 0,
        }
    ],
}

assert find_retaining_edge(graph) == "items[0]"
```

## What the gate checks

The gate builds several graphs containing real Python references. It computes the expected answer by creating a `weakref` to the target object, temporarily cutting each candidate edge, and checking whether the weak reference becomes dead.

The submitted implementation must return exactly the edge id selected by this CPython reference-counting oracle.
