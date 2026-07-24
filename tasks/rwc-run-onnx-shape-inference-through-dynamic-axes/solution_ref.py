from collections import Counter


def _add_dims(vals):
    """Symbolic sum of a list of int/str dims, canonicalised (see task.md)."""
    total = 0
    syms = []
    for v in vals:
        if isinstance(v, int):
            total += v
        else:
            syms.append(str(v))
    if not syms:
        return total
    counts = Counter(syms)
    parts = [f"{n}*{s}" if n > 1 else s for s, n in sorted(counts.items())]
    if total:
        parts.append(str(total))
    return "+".join(parts)


def infer_shapes(input_shapes: dict, graph: list) -> dict:
    """
    Static symbolic shape inference over a small ONNX-style graph.
    See task.md for the per-op rules (MatMul / Reshape / Concat / Gather).
    """
    shapes = {k: list(v) for k, v in input_shapes.items()}
    out = {}
    for node in graph:
        op = node["op"]
        name = node["name"]
        ins = node["inputs"]
        attrs = node.get("attrs", {})

        if op == "MatMul":
            a, b = shapes[ins[0]], shapes[ins[1]]
            if len(a) == 2 and len(b) == 2:
                shp = [a[0], b[1]]
            else:
                shp = [a[0], a[1], b[2]]

        elif op == "Reshape":
            a = shapes[ins[0]]
            target = attrs["shape"]
            shp = [a[i] if t == 0 else t for i, t in enumerate(target)]

        elif op == "Concat":
            axis = attrs["axis"]
            in_shapes = [shapes[n] for n in ins]
            rank = len(in_shapes[0])
            shp = []
            for d in range(rank):
                if d == axis:
                    shp.append(_add_dims([s[d] for s in in_shapes]))
                else:
                    shp.append(in_shapes[0][d])

        elif op == "Gather":
            axis = attrs["axis"]
            idx_shape = attrs["indices_shape"]
            data = shapes[ins[0]]
            shp = list(data[:axis]) + list(idx_shape) + list(data[axis + 1:])

        else:
            raise ValueError(f"unknown op {op}")

        shapes[name] = shp
        out[name] = shp
    return out
