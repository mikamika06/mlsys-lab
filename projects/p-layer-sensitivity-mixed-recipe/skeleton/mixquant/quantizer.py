def quantize_weight(w, bits):
    raise NotImplementedError

def run_forward(weights, bit_recipe, inputs):
    raise NotImplementedError

def evaluate_model(weights, bit_recipe, inputs, ref_output=None):
    raise NotImplementedError
