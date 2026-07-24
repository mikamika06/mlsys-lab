def _oracle(weight_shapes):
    n = len(weight_shapes)
    if n == 3:
        h0, in0 = weight_shapes[0]
        h1, in1 = weight_shapes[1]
        out2, in2 = weight_shapes[2]
        # GeGLU: first two layers produce the same hidden size
        if h0 != h1:
            raise ValueError("GeGLU gate and act dims differ")
        # Optional consistency check that second layer feeds into third
        if in2 != h0:
            raise ValueError("GeGLU hidden size mismatch with output layer")
        return "geglu"
    elif n == 2:
        h0, _ = weight_shapes[0]
        out1, in1 = weight_shapes[1]
        # Vanilla: first layer's output equals second layer's input
        if h0 == in1:
            return "vanilla"
        # SwiGLU: first layer's output is twice the second layer's input
        elif h0 == 2 * in1:
            return "swi_glu"
        else:
            raise ValueError("Unknown variant")
    else:
        raise ValueError("Unsupported number of layers")

def grade(sol, fx) -> dict:
    cases = [
        [(128, 64), (10, 128)],          # vanilla
        [(256, 64), (10, 128)],          # swi_glu
        [(128, 64), (128, 64), (10, 128)]# geglu
    ]
    ok = 1.0
    for shapes in cases:
        try:
            got = sol.classify_ffn_variant(shapes)
        except Exception:
            return {"exact_match": 0.0}
        try:
            expected = _oracle(shapes)
        except Exception:
            return {"exact_match": 0.0}
        if not isinstance(got, str):
            return {"exact_match": 0.0}
        if got != expected:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
