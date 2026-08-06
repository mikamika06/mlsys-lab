def track_shape_compilations(shape_sequence, dynamic=False, base_compile_cost=10.0, dynamic_compile_cost=25.0):
    """Track cumulative compile overhead as shape sequence expands."""
    steps = []
    cum_cost = 0.0
    recompilations = 0

    if not dynamic:
        compiled = set()
        for shape in shape_sequence:
            shape_tuple = tuple(shape)
            if shape_tuple not in compiled:
                compiled.add(shape_tuple)
                recompiled = True
                overhead = float(base_compile_cost)
                recompilations += 1
            else:
                recompiled = False
                overhead = 0.0
            cum_cost += overhead
            steps.append({
                "shape": shape_tuple,
                "recompiled": recompiled,
                "overhead": overhead,
                "cum_overhead": cum_cost
            })
        return {
            "steps": steps,
            "total_recompilations": recompilations,
            "total_overhead": cum_cost,
            "is_generalized": False
        }

    specialized = set()
    is_generalized = False
    for shape in shape_sequence:
        shape_tuple = tuple(shape)
        if is_generalized:
            recompiled = False
            overhead = 0.0
        elif shape_tuple in specialized:
            recompiled = False
            overhead = 0.0
        elif len(specialized) < 2:
            specialized.add(shape_tuple)
            recompiled = True
            overhead = float(base_compile_cost)
            recompilations += 1
        else:
            is_generalized = True
            recompiled = True
            overhead = float(dynamic_compile_cost)
            recompilations += 1

        cum_cost += overhead
        steps.append({
            "shape": shape_tuple,
            "recompiled": recompiled,
            "overhead": overhead,
            "cum_overhead": cum_cost
        })

    return {
        "steps": steps,
        "total_recompilations": recompilations,
        "total_overhead": cum_cost,
        "is_generalized": is_generalized
    }
