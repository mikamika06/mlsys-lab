def collect_layer_outputs(fp16_dict, int8_dict):
    raise NotImplementedError

def compute_layer_mse(fp16_dict, int8_dict):
    raise NotImplementedError

def identify_sensitive_layers(fp16_dict, int8_dict, top_k=2):
    raise NotImplementedError
