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
