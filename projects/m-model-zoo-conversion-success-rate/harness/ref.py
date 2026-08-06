MODELS = [
    {
        "name": "model_alpha",
        "signature_bytes": b"SIG:default|INPUT:x:float32[1,3,224,224]|OUTPUT:out:float32[1,1000]",
        "errors": ["OpCode not supported: CUSTOM_OP", "OutOfMemoryError: serialization failed"]
    },
    {
        "name": "model_beta",
        "signature_bytes": b"SIG:predict|INPUT:input_tensor:int32[10]|OUTPUT:logits:float32[10,2]",
        "errors": ["ValueError: dynamic dimensions not allowed"]
    },
    {
        "name": "model_gamma",
        "signature_bytes": b"SIG:encode|INPUT:tokens:int64[1,512]|OUTPUT:embeddings:float32[1,512,768]",
        "errors": ["KeyError: Missing tensor info"]
    }
]

def read_signature(data):
    text = data.decode("utf-8")
    parts = text.split("|")
    sig_name = parts[0].split(":")[1]
    inputs = {}
    outputs = {}
    for p in parts[1:]:
        key, val = p.split(":")
        if key == "INPUT":
            name, rest = val.split("[")
            dtype, shape_str = name.split(":")
            shape = [int(x) for x in rest.rstrip("]").split(",") if x]
            inputs[dtype] = {"shape": shape, "dtype": shape_str}
        elif key == "OUTPUT":
            name, rest = val.split("[")
            dtype, shape_str = name.split(":")
            shape = [int(x) for x in rest.rstrip("]").split(",") if x]
            outputs[dtype] = {"shape": shape, "dtype": shape_str}
    return {"signature": sig_name, "inputs": inputs, "outputs": outputs}

def classify_error(err_str):
    if "OpCode not supported" in err_str:
        return "unsupported_op"
    if "dynamic dimensions" in err_str:
        return "dynamic_shape"
    if "OutOfMemoryError" in err_str:
        return "out_of_memory"
    if "Missing tensor" in err_str:
        return "missing_tensor"
    return "unknown"

def compute_success_rate(results):
    if not results:
        return 0.0
    successes = sum(1 for r in results if r.get("success", False))
    return float(successes) / float(len(results))
