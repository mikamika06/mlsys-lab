def insert_cast_nodes(graph, start_name, end_name):
    out = []
    for L in graph:
        if L["name"] == start_name:
            out.append({"name": f"{start_name}_in_cast", "op": "cast", "to": "float32"})
        out.append(L)
        if L["name"] == end_name:
            out.append({"name": f"{end_name}_out_cast", "op": "cast", "to": "float16"})
    return out
