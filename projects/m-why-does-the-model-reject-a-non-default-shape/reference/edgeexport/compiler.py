def compile_with_shapes(model, shape_list, max_compile_time):
    cost_per_shape = 12
    total_cost = len(shape_list) * cost_per_shape
    if total_cost > max_compile_time:
        return {"status": "rejected", "reason": "compile_time_exceeded", "shapes_count": len(shape_list)}
    return {"status": "compiled", "shapes_count": len(shape_list), "cost": total_cost}
