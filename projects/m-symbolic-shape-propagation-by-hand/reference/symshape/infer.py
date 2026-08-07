def _canonicalize(expr):
    if isinstance(expr, int):
        return str(expr)
    expr = str(expr).strip()
    return expr


def _expr_equal(a, b):
    return _canonicalize(a) == _canonicalize(b)


def _eval_dim(expr, env):
    if isinstance(expr, int):
        return expr
    s = str(expr).strip()
    if s.isdigit():
        return int(s)
    if s in env:
        return env[s]
    try:
        return int(eval(s, {"__builtins__": None}, env))
    except Exception:
        return s


def _broadcast_dims(d1, d2):
    if _expr_equal(d1, d2):
        return d1
    if str(d1) == "1":
        return d2
    if str(d2) == "1":
        return d1
    if isinstance(d1, int) and isinstance(d2, int):
        if d1 == d2:
            return d1
        if d1 == 1:
            return d2
        if d2 == 1:
            return d1
        return None
    return None


def _broadcast_shapes(s1, s2):
    if s1 is None or s2 is None:
        return None
    r1, r2 = len(s1), len(s2)
    max_r = max(r1, r2)
    p1 = [1] * (max_r - r1) + list(s1)
    p2 = [1] * (max_r - r2) + list(s2)
    out = []
    for d1, d2 in zip(p1, p2):
        b = _broadcast_dims(d1, d2)
        if b is None:
            return None
        out.append(b)
    return tuple(out)


def _infer_node(op, inputs, params):
    if op in ("Add", "Sub", "Mul", "Div"):
        if len(inputs) != 2:
            return None
        return _broadcast_shapes(inputs[0], inputs[1])

    if op == "MatMul":
        if len(inputs) != 2:
            return None
        s1, s2 = inputs[0], inputs[1]
        if s1 is None or s2 is None or len(s1) < 2 or len(s2) < 2:
            return None
        if not _expr_equal(s1[-1], s2[-2]):
            return None
        batch1, batch2 = s1[:-2], s2[:-2]
        batch_out = _broadcast_shapes(batch1, batch2)
        if batch_out is None:
            return None
        return tuple(list(batch_out) + [s1[-2], s2[-1]])

    if op == "Reshape":
        if len(inputs) < 1:
            return None
        in_shape = inputs[0]
        target = params.get("shape")
        if in_shape is None or target is None:
            return None
        target = list(target)
        neg_idx = [i for i, d in enumerate(target) if d == -1 or str(d) == "-1"]
        if len(neg_idx) > 1:
            return None
        if len(neg_idx) == 0:
            return tuple(target)
        known_prod = 1
        has_symbol = False
        for i, d in enumerate(target):
            if i in neg_idx:
                continue
            if isinstance(d, int) and d > 0:
                known_prod *= d
            elif str(d).isdigit() and int(d) > 0:
                known_prod *= int(d)
            else:
                has_symbol = True

        in_prod = 1
        in_has_symbol = False
        for d in in_shape:
            if isinstance(d, int):
                in_prod *= d
            elif str(d).isdigit():
                in_prod *= int(d)
            else:
                in_has_symbol = True

        if not has_symbol and not in_has_symbol and known_prod > 0:
            if in_prod % known_prod == 0:
                target[neg_idx[0]] = in_prod // known_prod
                return tuple(target)

        if len(in_shape) == 1 and len(target) == 2 and neg_idx[0] == 1 and target[0] == in_shape[0]:
            target[neg_idx[0]] = 1
            return tuple(target)

        return None

    if op == "Concat":
        if not inputs or any(s is None for s in inputs):
            return None
        axis = params.get("axis", 0)
        rank = len(inputs[0])
        if axis < 0:
            axis += rank
        if not (0 <= axis < rank):
            return None
        for s in inputs[1:]:
            if len(s) != rank:
                return None
            for i in range(rank):
                if i != axis and not _expr_equal(s[i], inputs[0][i]):
                    return None
        out_shape = list(inputs[0])
        concat_dims = [s[axis] for s in inputs]
        if all(isinstance(d, int) for d in concat_dims):
            out_shape[axis] = sum(concat_dims)
        else:
            out_shape[axis] = " + ".join(str(d) for d in concat_dims)
        return tuple(out_shape)

    if op == "Transpose":
        if len(inputs) != 1 or inputs[0] is None:
            return None
        perm = params.get("perm")
        in_shape = inputs[0]
        if perm is None:
            return tuple(reversed(in_shape))
        if len(perm) != len(in_shape):
            return None
        return tuple(in_shape[p] for p in perm)

    return None


def propagate_shapes(graph):
    shapes = dict(graph.get("inputs", {}))
    nodes = graph.get("nodes", [])

    changed = True
    while changed:
        changed = False
        for node in nodes:
            name = node["name"]
            if name in shapes and shapes[name] is not None:
                continue
            in_shapes = [shapes.get(inp) for inp in node.get("inputs", [])]
            if any(s is None for s in in_shapes):
                continue
            out_shape = _infer_node(node["op"], in_shapes, node.get("params", {}))
            if out_shape is not None:
                shapes[name] = out_shape
                changed = True

    return shapes


def find_first_failure(graph):
    shapes = dict(graph.get("inputs", {}))
    nodes = graph.get("nodes", [])

    for node in nodes:
        in_shapes = [shapes.get(inp) for inp in node.get("inputs", [])]
        if any(s is None for s in in_shapes):
            return node["name"]
        out_shape = _infer_node(node["op"], in_shapes, node.get("params", {}))
        if out_shape is None:
            return node["name"]
        shapes[node["name"]] = out_shape

    return None
