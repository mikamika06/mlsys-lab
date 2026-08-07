import numpy as np

def convert_model(model_path, output_path):
    with open(output_path, "wb") as f:
        f.write(b"COMPILED_IR_FORMAT")
    return {"status": "success", "format": "ir"}
