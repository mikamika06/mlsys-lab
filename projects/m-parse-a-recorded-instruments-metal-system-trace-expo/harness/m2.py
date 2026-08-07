import ref


def check(workdir):
    from mpslab.graphs import compare_execution

    try:
        res_loop = compare_execution("loop", 50)
        res_graph = compare_execution("graph", 50)
    except Exception as e:
        return {"graph_compared": 0.0, "_note": f"compare_execution raised {type(e).__name__}: {e}"}

    want_loop = ref.count_command_buffers("loop", 50)
    want_graph = ref.count_command_buffers("graph", 50)

    if res_loop == want_loop and res_graph == want_graph:
        return {"graph_compared": 1.0}
    return {"graph_compared": 0.0, "_note": f"loop got {res_loop} (want {want_loop}), graph got {res_graph} (want {want_graph})"}
