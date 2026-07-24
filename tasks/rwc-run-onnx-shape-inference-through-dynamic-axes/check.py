import numpy as np
from collections import Counter

_SYMBOLS = ["N", "M", "B", "K", "T", "D", "S", "L"]


def _add_dims(vals):
    total = 0
    syms = []
    for v in vals:
        if isinstance(v, (int, np.integer)):
            total += int(v)
        else:
            syms.append(str(v))
    if not syms:
        return total
    counts = Counter(syms)
    parts = [f"{n}*{s}" if n > 1 else s for s, n in sorted(counts.items())]
    if total:
        parts.append(str(total))
    return "+".join(parts)


def _oracle_infer(input_shapes, graph):
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
            raise ValueError(op)

        shapes[name] = shp
        out[name] = shp
    return out


def _sym_pool(rng, n):
    perm = rng.permutation(len(_SYMBOLS))
    return [_SYMBOLS[i] for i in perm[:n]]


def _build_cases(rng):
    cases = []

    # Template A: MatMul -> Concat(axis=1, batch untouched) -> Reshape(copy via 0)
    # -> Gather(axis=0, replaces the dynamic batch axis).
    for _ in range(3):
        sym_batch = _sym_pool(rng, 1)[0]
        dim_a = int(rng.integers(2, 6))
        dim_b = int(rng.integers(2, 6))
        dim_c = int(rng.integers(2, 6))
        idx_dim = int(rng.integers(1, 5))
        input_shapes = {
            "x": [sym_batch, dim_a],
            "w": [dim_a, dim_b],
            "y": [sym_batch, dim_c],
        }
        graph = [
            {"name": "mm_out", "op": "MatMul", "inputs": ["x", "w"], "attrs": {}},
            {"name": "concat_out", "op": "Concat", "inputs": ["mm_out", "y"], "attrs": {"axis": 1}},
            {"name": "reshape_out", "op": "Reshape", "inputs": ["concat_out"],
             "attrs": {"shape": [0, dim_b + dim_c]}},
            {"name": "gather_out", "op": "Gather", "inputs": ["reshape_out"],
             "attrs": {"axis": 0, "indices_shape": [idx_dim]}},
        ]
        cases.append((input_shapes, graph))

    # Template B: two distinct symbols concatenated along axis 0 -> "sym1+sym2",
    # a Reshape that copies both axes via 0, then a rank-changing Gather.
    for _ in range(3):
        sym1, sym2 = _sym_pool(rng, 2)
        d = int(rng.integers(2, 6))
        i1 = int(rng.integers(1, 4))
        i2 = int(rng.integers(1, 4))
        input_shapes = {"p": [sym1, d], "q": [sym2, d]}
        graph = [
            {"name": "concat_out", "op": "Concat", "inputs": ["p", "q"], "attrs": {"axis": 0}},
            {"name": "reshape_out", "op": "Reshape", "inputs": ["concat_out"], "attrs": {"shape": [0, 0]}},
            {"name": "gather_out", "op": "Gather", "inputs": ["reshape_out"],
             "attrs": {"axis": 1, "indices_shape": [i1, i2]}},
        ]
        cases.append((input_shapes, graph))

    # Template C: the same symbol repeated across 3 concat inputs -> "3*sym"
    # coefficient collapsing, mixed with a plain int axis elsewhere.
    for _ in range(3):
        sym = _sym_pool(rng, 1)[0]
        d = int(rng.integers(2, 6))
        idx = int(rng.integers(1, 4))
        input_shapes = {"u": [sym, d], "v": [sym, d], "w2": [sym, d]}
        graph = [
            {"name": "concat_out", "op": "Concat", "inputs": ["u", "v", "w2"], "attrs": {"axis": 0}},
            {"name": "gather_out", "op": "Gather", "inputs": ["concat_out"],
             "attrs": {"axis": 1, "indices_shape": [idx]}},
        ]
        cases.append((input_shapes, graph))

    # Template D: a mix of concrete ints and one repeated symbol on the
    # concat axis -> exercises "n*sym+total".
    for _ in range(2):
        sym = _sym_pool(rng, 1)[0]
        d = int(rng.integers(2, 6))
        c1 = int(rng.integers(1, 5))
        c2 = int(rng.integers(1, 5))
        input_shapes = {"a1": [sym, d], "a2": [c1, d], "a3": [sym, d], "a4": [c2, d]}
        graph = [
            {"name": "concat_out", "op": "Concat", "inputs": ["a1", "a2", "a3", "a4"], "attrs": {"axis": 0}},
        ]
        cases.append((input_shapes, graph))

    return cases


def _dim_eq(a, b):
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def grade(sol, fx) -> dict:
    """
    Builds several small ONNX-style graphs mixing MatMul/Reshape/Concat/Gather
    with dynamic (symbolic) axes and compares the student's inferred shapes
    for every node, dimension by dimension, against a reference implementation
    of the canonical shape-inference rules described in task.md.
    """
    rng = np.random.default_rng(0)
    cases = _build_cases(rng)
    ok = 1.0
    for input_shapes, graph in cases:
        expected = _oracle_infer(input_shapes, graph)
        try:
            got = sol.infer_shapes(dict(input_shapes), [dict(n) for n in graph])
        except Exception:
            ok = 0.0
            break

        if not isinstance(got, dict):
            ok = 0.0
            break

        bad = False
        for name, shp in expected.items():
            if name not in got:
                bad = True
                break
            g = got[name]
            try:
                if len(g) != len(shp):
                    bad = True
                    break
                for a, b in zip(g, shp):
                    if not _dim_eq(a, b):
                        bad = True
                        break
            except Exception:
                bad = True
            if bad:
                break
        if bad:
            ok = 0.0
            break
    return {"exact_match": ok}
