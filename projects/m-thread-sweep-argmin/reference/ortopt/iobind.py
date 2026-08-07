import numpy as np

def run_with_iobinding(mock_session, inputs):
    outputs = []
    for inp in inputs:
        out = inp * 2.0
        outputs.append(out)
    return outputs
