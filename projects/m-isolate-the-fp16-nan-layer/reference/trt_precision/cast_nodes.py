def build_strongly_typed_islands(graph, sensitive_layer_names):
    nodes = graph["nodes"]
    new_nodes = []
    cast_counter = 0

    for node in nodes:
        if node["name"] in sensitive_layer_names:
            in_var = node["inputs"][0]
            fp32_in_var = f"{in_var}_fp32_{cast_counter}"
            cast_in = {
                "name": f"Cast_to_fp32_{cast_counter}",
                "op": "Cast",
                "inputs": [in_var],
                "outputs": [fp32_in_var],
                "dtype": "FLOAT",
                "to": "FLOAT"
            }
            new_nodes.append(cast_in)

            out_var = node["outputs"][0]
            fp32_out_var = f"{out_var}_fp32_{cast_counter}"
            island_node = dict(node)
            island_node["inputs"] = [fp32_in_var]
            island_node["outputs"] = [fp32_out_var]
            island_node["dtype"] = "FLOAT"
            island_node["precision_mode"] = "strongly_typed"
            new_nodes.append(island_node)

            cast_out = {
                "name": f"Cast_to_fp16_{cast_counter}",
                "op": "Cast",
                "inputs": [fp32_out_var],
                "outputs": [out_var],
                "dtype": "FLOAT16",
                "to": "FLOAT16"
            }
            new_nodes.append(cast_out)
            cast_counter += 1
        else:
            mod_node = dict(node)
            mod_node["dtype"] = "FLOAT16"
            mod_node["precision_mode"] = "strongly_typed"
            new_nodes.append(mod_node)

    return {"nodes": new_nodes, "mode": "strongly_typed"}
