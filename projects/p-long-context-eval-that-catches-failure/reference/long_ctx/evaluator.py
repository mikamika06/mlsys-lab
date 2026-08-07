def evaluate_position_curve(model, contexts) -> dict:
    results = {}
    for pos, ctx in contexts.items():
        pred = model(ctx)
        results[pos] = 1.0 if "SECRET_FACT" in pred else 0.0
    return results
