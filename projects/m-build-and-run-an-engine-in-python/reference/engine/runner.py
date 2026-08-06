import numpy as np

def execute_engine(engine, inputs):
    config = engine["config"]
    max_batch = config.get("max_batch_size", 1)
    outputs = []
    for inp in inputs:
        if inp.shape[0] > max_batch:
            raise ValueError("Batch size exceeds maximum configured batch size")
        out = inp * 2.0 if not config.get("fp16") else inp * 1.5
        outputs.append(out)
    return outputs
