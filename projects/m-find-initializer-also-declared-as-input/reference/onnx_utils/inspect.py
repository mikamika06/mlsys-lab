def find_initializer_inputs(model_proto):
    init_names = {init.name for init in model_proto.graph.initializer}
    input_names = {inp.name for inp in model_proto.graph.input}
    return sorted(list(init_names.intersection(input_names)))
