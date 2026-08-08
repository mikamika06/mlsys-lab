GRAPHS = [
    [
        {"op": "placeholder", "target": "x"},
        {"op": "call_function", "target": "aten.linear.default"},
        {"op": "call_function", "target": "aten.relu.default"},
        {"op": "output", "target": "output"}
    ],
    [
        {"op": "placeholder", "target": "x"},
        {"op": "call_function", "target": "torch.ops.aten.add.Tensor"},
        {"op": "call_function", "target": "aten.native_layer_norm.default"},
        {"op": "call_module", "target": "submod"},
        {"op": "output", "target": "output"}
    ],
    [
        {"op": "placeholder", "target": "x"},
        {"op": "call_function", "target": "torch.ops.higher_order.cond"},
        {"op": "call_function", "target": "aten.add.Tensor"},
        {"op": "call_function", "target": "aten.add.Tensor"},
        {"op": "output", "target": "output"}
    ]
]


class ExportError(Exception):
    pass


def dummy_export(model_def):
    if model_def.get("mutates"):
        raise ExportError(f"Unsupported global state mutation on '{model_def['mutates']}'")
    return "ExportedProgramMock"


MODELS = [
    {"name": "simple", "mutates": None},
    {"name": "counter", "mutates": "self.step_count"},
    {"name": "ema", "mutates": "self.moving_average"}
]
