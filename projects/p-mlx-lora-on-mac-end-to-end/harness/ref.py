import numpy as np
from lora_pipe.engine import prepare_data, run_lora, merge_adapter, quantize_model, LoraServer, evaluate_quality

def get_sample_data():
    return [{"prompt": "hello", "response": "world"}, {"prompt": "ml", "response": "systems"}]

def get_base_weights():
    return {"layer1": np.ones((8, 8)), "layer2": np.zeros((8, 8))}

def get_eval_set():
    return [{"prompt": "test1"}, {"prompt": "test2"}]
