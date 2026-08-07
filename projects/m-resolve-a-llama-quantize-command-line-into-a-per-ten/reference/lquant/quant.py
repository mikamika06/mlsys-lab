BYTES_PER_ELEM = {
    "F32": 4.0,
    "F16": 2.0,
    "Q8_0": 1.0,
    "Q4_0": 0.5,
}

def predict_size(model_tensors, plan):
    total = 0
    for name, shape in model_tensors.items():
        num_elems = 1
        for dim in shape:
            num_elems *= dim
        ttype = plan.get(name, "F32")
        bpe = BYTES_PER_ELEM.get(ttype, 4.0)
        total += int(num_elems * bpe)
    return total
