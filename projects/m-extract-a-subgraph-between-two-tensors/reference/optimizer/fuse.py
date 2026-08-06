import onnx


def fuse_gelu(model):
    graph = model.graph
    nodes = list(graph.node)

    output_to_node = {out: n for n in nodes for out in n.output}

    fused_nodes = []
    skip_nodes = set()

    for node in nodes:
        if node.name in skip_nodes:
            continue

        if node.op_type == "Mul" and len(node.input) == 2:
            mul1_in = node.input

            erf_node = None
            other_input = None
            for inp in mul1_in:
                candidate = output_to_node.get(inp)
                if candidate and candidate.op_type == "Erf":
                    erf_node = candidate
                else:
                    other_input = inp

            if erf_node and other_input:
                add_node = output_to_node.get(erf_node.input[0])
                if add_node and add_node.op_type == "Add":
                    mul2_node = None
                    for add_inp in add_node.input:
                        cand = output_to_node.get(add_inp)
                        if cand and cand.op_type == "Mul":
                            mul2_node = cand
                            break
                    if mul2_node:
                        skip_nodes.add(node.name)
                        skip_nodes.add(erf_node.name)
                        skip_nodes.add(add_node.name)
                        skip_nodes.add(mul2_node.name)

                        gelu_node = onnx.helper.make_node(
                            "Gelu",
                            inputs=[mul2_node.input[0] if mul2_node.input[0] != add_node.name else mul2_node.input[1]],
                            outputs=node.output,
                            name=f"FusedGelu_{node.name}"
                        )
                        fused_nodes.append(gelu_node)
                        continue

        fused_nodes.append(node)

    del graph.node[:]
    graph.node.extend(fused_nodes)
    return model
