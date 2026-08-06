def optimize_graph(graph, mode):
    if mode == "transformers":
        return graph + "_opt_trans"
    elif mode == "all":
        return graph + "_opt_all"
    return graph
