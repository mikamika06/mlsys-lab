def classify_ops():
    return {
        "aten.mm": "fp16",
        "aten.bmm": "fp16",
        "aten.addmm": "fp16",
        "aten.conv2d": "fp16",
        "aten.linear": "fp16",
        "aten.sum": "promote",
        "aten.mean": "promote",
        "aten.softmax": "promote",
        "aten.layer_norm": "promote",
        "aten.gelu": "promote",
        "aten.pow": "promote",
        "aten.div": "promote",
        "aten.exp": "promote",
        "aten.log": "promote",
        "aten.sin": "promote"
    }
