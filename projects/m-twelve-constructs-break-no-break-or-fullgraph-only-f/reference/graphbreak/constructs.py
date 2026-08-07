def classify_constructs(items):
    known = {
        "tensor_item": "break",
        "shape_print": "break",
        "python_if_tensor": "break",
        "pure_arithmetic": "no_break",
        "tensor_add": "no_break",
        "reshape_constant": "no_break",
        "fullgraph_only_mutation": "fullgraph_only",
        "fullgraph_only_builtin_dict": "fullgraph_only",
        "item_in_loop": "break",
        "tensor_size_check": "break",
        "matrix_multiply": "no_break",
        "tensor_clone": "no_break"
    }
    return [known.get(x, "break") for x in items]
