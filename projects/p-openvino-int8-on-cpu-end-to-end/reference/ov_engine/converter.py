import os
import numpy as np

def convert_model(model_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"SIMULATED_OV_MODEL_IR")
    return True
