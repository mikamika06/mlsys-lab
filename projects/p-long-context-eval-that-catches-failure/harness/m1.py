from long_ctx.generator import generate_context_with_fact

def check(workdir):
    m = {"positions_covered": 0.0, "fact_retrievable": 0.0}
    try:
        ctx1 = generate_context_with_fact(500, 0.1, "SECRET_FACT")
        ctx2 = generate_context_with_fact(500, 0.5, "SECRET_FACT")
        ctx3 = generate_context_with_fact(500, 0.9, "SECRET_FACT")
        if ctx1 and ctx2 and ctx3:
            m["positions_covered"] = 1.0
        if "SECRET_FACT" in ctx1 and "SECRET_FACT" in ctx2 and "SECRET_FACT" in ctx3:
            m["fact_retrievable"] = 1.0
    except Exception:
        pass
    return m
