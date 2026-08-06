import jax


def run_with_mutable_closure(f, x):
    leaked = []

    @jax.jit
    def wrapped(val):
        res = f(val)
        leaked.append(res)
        return res

    try:
        out = wrapped(x)
        jax.block_until_ready(out)
        return "safe", out
    except Exception as e:
        if "LeakedTracer" in type(e).__name__ or "Tracer" in str(e):
            return "leaked", leaked
        raise e
