SNIPPETS = [
    {
        "id": "snippet_01",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": False
    },
    {
        "id": "snippet_02",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": True,
        "has_invalid_shape_or_type": False
    },
    {
        "id": "snippet_03",
        "has_data_dependent_branch": True,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": False
    },
    {
        "id": "snippet_04",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": True
    },
    {
        "id": "snippet_05",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": False
    },
    {
        "id": "snippet_06",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": True,
        "has_invalid_shape_or_type": False
    },
    {
        "id": "snippet_07",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": True
    },
    {
        "id": "snippet_08",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": False
    },
    {
        "id": "snippet_09",
        "has_data_dependent_branch": True,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": False
    },
    {
        "id": "snippet_10",
        "has_data_dependent_branch": False,
        "has_unsupported_side_effect": False,
        "has_invalid_shape_or_type": False
    }
]

SHAPE_SEQUENCE = [
    (1, 16),
    (1, 32),
    (1, 16),
    (1, 64),
    (1, 128),
    (1, 32),
    (1, 256)
]

TRACE_TIMINGS = {
    "raw_eager": [0.012, 0.012, 0.011, 0.012, 0.012],
    "aot_eager": [0.150, 0.012, 0.012, 0.011, 0.012],
    "inductor": [1.850, 0.004, 0.004, 0.004, 0.004]
}


def classify_snippets(snippets):
    results = []
    for s in snippets:
        snip_id = s["id"]
        invalid = s.get("has_invalid_shape_or_type", False)
        break_cond = s.get("has_data_dependent_branch", False) or s.get("has_unsupported_side_effect", False)

        if invalid:
            default_res = "error"
            fullgraph_res = "error"
        elif break_cond:
            default_res = "graph_break"
            fullgraph_res = "error"
        else:
            default_res = "pass"
            fullgraph_res = "pass"

        results.append({
            "id": snip_id,
            "default": default_res,
            "fullgraph": fullgraph_res
        })
    return results


def track_shape_compilations(shape_sequence, dynamic=False, base_compile_cost=10.0, dynamic_compile_cost=25.0):
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


def isolate_backend_overhead(timings):
    raw_eager = timings["raw_eager"]
    aot_eager = timings["aot_eager"]
    inductor = timings["inductor"]

    raw_mean = sum(raw_eager) / float(len(raw_eager))

    aot_first = float(aot_eager[0])
    aot_warm_mean = sum(aot_eager[1:]) / float(len(aot_eager[1:]))
    tracing_overhead = max(0.0, aot_first - aot_warm_mean)

    ind_first = float(inductor[0])
    ind_warm_mean = sum(inductor[1:]) / float(len(inductor[1:]))
    total_ind_overhead = max(0.0, ind_first - ind_warm_mean)

    ind_compile_only = max(0.0, total_ind_overhead - tracing_overhead)
    speedup = raw_mean / ind_warm_mean if ind_warm_mean > 0 else 1.0

    return {
        "tracing_overhead": round(tracing_overhead, 6),
        "total_inductor_overhead": round(total_ind_overhead, 6),
        "inductor_compile_only_overhead": round(ind_compile_only, 6),
        "speedup_ratio": round(speedup, 6)
    }
