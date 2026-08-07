def simulate_recompile_storm(shapes, cache_size_limit):
    compiled_shapes = set()
    history = []
    fallback_step = None
    for idx, shape in enumerate(shapes):
        if shape in compiled_shapes:
            status = "hit"
        elif len(compiled_shapes) < cache_size_limit:
            status = "recompile"
            compiled_shapes.add(shape)
        else:
            status = "eager_fallback"
            if fallback_step is None:
                fallback_step = idx
        history.append({"step": idx, "shape": shape, "status": status})
    return {
        "history": history,
        "fallback_step": fallback_step,
        "total_recompiles": len(compiled_shapes),
        "total_fallbacks": sum(1 for h in history if h["status"] == "eager_fallback")
    }
