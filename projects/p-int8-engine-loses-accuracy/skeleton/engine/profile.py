def profile_layers(model, fp16_outputs, int8_outputs):
    raise NotImplementedError

def identify_sensitive_layers(layer_mses, top_k=3):
    raise NotImplementedError
