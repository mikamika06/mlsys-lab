def confirm_relationship(constraints, dim2, dim1):
    def get_base(d):
        k_total = 1
        curr = d
        while curr in constraints:
            k, nxt = constraints[curr]
            k_total *= k
            curr = nxt
        return k_total, curr

    k2, b2 = get_base(dim2)
    k1, b1 = get_base(dim1)
    if b2 == b1 and k2 % k1 == 0:
        return k2 // k1
    return None

def propagate_shapes(ops, inputs, constraints):
    def get_base(d):
        if d is None:
            return 1, None
        k_total = 1
        curr = d
        while curr in constraints:
            k, nxt = constraints[curr]
            k_total *= k
            curr = nxt
        return k_total, curr

    out = {}
    for name, shape in inputs.items():
        base_shape = []
        for k, var in shape:
            kb, b = get_base(var)
            base_shape.append((k * kb, b))
        out[name] = tuple(base_shape)

    for op in ops:
        in_name = op["in"]
        out_name = op["out"]
        tgt_shape = op["shape"]

        in_shape = out[in_name]

        in_coef = 1
        in_vars = {}
        for k, var in in_shape:
            in_coef *= k
            if var is not None:
                in_vars[var] = in_vars.get(var, 0) + 1

        tgt_base = []
        tgt_coef = 1
        tgt_vars = {}
        neg_idx = -1
        for i, (k, var) in enumerate(tgt_shape):
            if k == -1:
                neg_idx = i
                tgt_base.append(None)
            else:
                kb, b = get_base(var)
                tgt_base.append((k * kb, b))
                tgt_coef *= (k * kb)
                if b is not None:
                    tgt_vars[b] = tgt_vars.get(b, 0) + 1

        if neg_idx != -1:
            res_coef = in_coef // tgt_coef
            res_vars = {}
            for v, count in in_vars.items():
                diff = count - tgt_vars.get(v, 0)
                if diff > 0:
                    res_vars[v] = diff

            if not res_vars:
                tgt_base[neg_idx] = (res_coef, None)
            else:
                v = list(res_vars.keys())[0]
                tgt_base[neg_idx] = (res_coef, v)

        out[out_name] = tuple(tgt_base)
    return out
