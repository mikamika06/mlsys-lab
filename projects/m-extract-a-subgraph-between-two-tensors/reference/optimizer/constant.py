import onnx


def fold_initializer_constants(model):
    graph = model.graph
    used_inputs = set()
    for node in graph.node:
        for inp in node.input:
            used_inputs.add(inp)

    new_initializers = [init for init in graph.initializer if init.name in used_inputs]
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    return model
